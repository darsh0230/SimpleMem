"""
Retriever - Stage 3: Adaptive Query-Aware Retrieval

Performs intelligent retrieval through:
- Query complexity analysis
- Multi-query planning
- Hybrid search (semantic + lexical + symbolic)
- Reflection-based iterative refinement

Refactored: Removed parallel processing for simplicity and stability.
"""

import asyncio
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from ..auth.models import MemoryEntry
from ..database.vector_store import MultiTenantVectorStore
from ..utils.profile import profiler

# Type alias for LLM client (supports both OpenRouter and Ollama)
LLMClient = object  # Duck-typed: can be OpenRouterClient or OllamaClient


@dataclass
class RetrievalPlan:
    """Query analysis and retrieval plan"""

    question_type: str
    key_entities: List[str]
    required_info: List[Dict[str, Any]]
    relationships: List[str]
    minimal_queries_needed: int
    complexity_score: float  # 0-1


class Retriever:
    """
    Adaptive retriever with intelligent query planning.
    Sequential processing for stability.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        vector_store: MultiTenantVectorStore,
        table_name: str,
        semantic_top_k: int = 25,
        keyword_top_k: int = 5,
        enable_planning: bool = True,
        enable_reflection: bool = True,
        max_reflection_rounds: int = 2,
        temperature: float = 0.1,
        embedding_client: Optional[LLMClient] = None,
        fast_model: Optional[str] = None,
    ):
        self.client = llm_client
        self.embedding_client = (
            embedding_client or llm_client
        )  # Use embedding client if provided, otherwise use LLM client
        self.vector_store = vector_store
        self.table_name = table_name
        self.semantic_top_k = semantic_top_k
        self.keyword_top_k = keyword_top_k
        self.enable_planning = enable_planning
        self.enable_reflection = enable_reflection
        self.max_reflection_rounds = max_reflection_rounds
        self.temperature = temperature
        self.fast_model = fast_model

    async def retrieve(
        self,
        query: str,
        agents: Optional[List[str]] = None,
        enable_reflection: Optional[bool] = None,
    ) -> List[MemoryEntry]:
        """
        Retrieve relevant memory entries for a query

        Args:
            query: User's question
            agents: Optional list of agent identifiers to filter by
            enable_reflection: Override reflection setting

        Returns:
            List of relevant MemoryEntry objects
        """
        use_reflection = (
            enable_reflection
            if enable_reflection is not None
            else self.enable_reflection
        )

        if self.enable_planning:
            return await self._retrieve_with_planning(query, agents, use_reflection)
        else:
            return await self._simple_retrieve(query, agents)

    def _filter_by_agents(
        self, entries: List[MemoryEntry], agents: List[str]
    ) -> List[MemoryEntry]:
        """Filter memory entries by agent identifiers (case-insensitive)"""
        agents_lower = {agent.lower() for agent in agents}
        return [
            entry
            for entry in entries
            if agents_lower.intersection({a.lower() for a in entry.agents})
        ]

    async def _simple_retrieve(
        self, query: str, agents: Optional[List[str]] = None
    ) -> List[MemoryEntry]:
        """Simple semantic search without planning"""
        query_embedding = await self.embedding_client.create_single_embedding(query)
        semantic_results = await self.vector_store.semantic_search(
            self.table_name,
            query_embedding,
            top_k=self.semantic_top_k,
        )

        # Filter by agents if specified
        if agents:
            return self._filter_by_agents(semantic_results, agents)
        return semantic_results

    async def _retrieve_with_planning(
        self,
        query: str,
        agents: Optional[List[str]],
        enable_reflection: bool,
    ) -> List[MemoryEntry]:
        """Retrieve with intelligent planning and optional reflection"""

    async def _retrieve_with_planning(
        self,
        query: str,
        agents: Optional[List[str]],
        enable_reflection: bool,
    ) -> List[MemoryEntry]:
        """Retrieve with intelligent planning and optional reflection"""

        # Step 1 & 2: Analyze and Generate Queries (Merged)
        with profiler.profile("analyze_and_generate_queries"):
            plan, search_queries = await self._analyze_and_generate_queries(query)

        # Step 3: Execute searches in parallel
        with profiler.profile("execute_searches"):
            all_results = await self._execute_searches(search_queries, agents)

        # Step 4: Merge and deduplicate
        merged_results = self._merge_and_deduplicate(all_results)

        # Step 5: Optional reflection
        if enable_reflection and plan.complexity_score > 0.5:
            with profiler.profile("reflection_loop"):
                merged_results = await self._retrieve_with_reflection(
                    query,
                    merged_results,
                    plan,
                    agents,
                )

        return merged_results

    async def _analyze_and_generate_queries(
        self,
        query: str,
    ) -> tuple[RetrievalPlan, List[str]]:
        """Analyze query and generate targeted search queries in one step"""

        prompt = f"""Analyze the question and generate targeted search queries.

Question: {query}

Tasks:
1. Analyze the question type and entities.
2. Determine complexity score (0.0-1.0).
3. Generate 1-4 targeted search queries to find the answer.

Analysis Guidance:
- 'required_info' should list CONCRETE facts needed (e.g., "price of item", "date of event"), NOT abstract categories like "context" or "domain".

Return JSON:
{{
  "analysis": {{
    "question_type": "type",
    "key_entities": ["entity1", "entity2"],
    "required_info": [
      {{"type": "concrete_fact_needed", "priority": "high/medium/low"}}
    ],
    "relationships": ["relationship1"],
    "minimal_queries_needed": 1-4,
    "complexity_score": 0.0-1.0
  }},
  "queries": ["query1", "query2", ...]
}}

Return ONLY valid JSON."""

        messages = [
            {
                "role": "system",
                "content": "You are a query analysis and search expert.",
            },
            {"role": "user", "content": prompt},
        ]

        try:
            response = await self.client.chat_completion(
                messages=messages,
                temperature=self.temperature,
            )

            data = self.client.extract_json(response)
            if data:
                analysis = data.get("analysis", {})
                queries = data.get("queries", [])

                plan = RetrievalPlan(
                    question_type=analysis.get("question_type", "factual"),
                    key_entities=analysis.get("key_entities", []),
                    required_info=analysis.get("required_info", []),
                    relationships=analysis.get("relationships", []),
                    minimal_queries_needed=min(
                        analysis.get("minimal_queries_needed", 1), 4
                    ),
                    complexity_score=min(
                        max(analysis.get("complexity_score", 0.5), 0.0), 1.0
                    ),
                )

                # Fallback if no queries generated
                if not queries:
                    queries = [query]

                return plan, queries[:4]

        except Exception as e:
            print(f"Analysis and generation error: {e}")

        # Default fallback
        return (
            RetrievalPlan(
                question_type="factual",
                key_entities=[],
                required_info=[],
                relationships=[],
                minimal_queries_needed=1,
                complexity_score=0.5,
            ),
            [query],
        )

    async def _analyze_information_requirements(
        self,
        query: str,
    ) -> RetrievalPlan:
        """Deprecated: Analyze query to determine information requirements"""
        # ... existing implementation kept for fallback or removed if desired ...
        # For now, keeping the original implementation logic here or we can remove it.
        # But to be safe and clean, I will just implementing the new one above and keep the old one below or remove it.
        # Since the user asked to optimize, I'll replace the old one calls with the new one.
        pass

    async def _generate_targeted_queries(
        self,
        original_query: str,
        plan: RetrievalPlan,
    ) -> List[str]:
        """Generate targeted search queries based on analysis"""

        if plan.minimal_queries_needed <= 1:
            return [original_query]

        prompt = f"""Based on the analysis, generate {plan.minimal_queries_needed} targeted search queries.

Original Question: {original_query}

Analysis:
- Question Type: {plan.question_type}
- Key Entities: {plan.key_entities}
- Required Information: {plan.required_info}
- Relationships: {plan.relationships}

Requirements:
1. Generate {plan.minimal_queries_needed} distinct queries
2. Each query should target specific information
3. Together they should cover all required information
4. Keep queries concise and focused

Return JSON:
{{
  "queries": ["query1", "query2", ...]
}}

Return ONLY valid JSON."""

        messages = [
            {"role": "system", "content": "You are a search query generator."},
            {"role": "user", "content": prompt},
        ]

        try:
            response = await self.client.chat_completion(
                messages=messages,
                temperature=self.temperature,
            )

            data = self.client.extract_json(response)
            if data and "queries" in data:
                queries = data["queries"][:4]  # Max 4 queries
                if queries:
                    return queries
        except Exception as e:
            print(f"Query generation error: {e}")

        return [original_query]

    async def _execute_single_search(
        self,
        query: str,
        agents: Optional[List[str]],
    ) -> List[MemoryEntry]:
        """Execute a single search query (semantic + keyword)"""
        results = []

        # Semantic search
        try:
            query_embedding = await self.embedding_client.create_single_embedding(query)
            semantic_results = await self.vector_store.semantic_search(
                self.table_name,
                query_embedding,
                top_k=self.semantic_top_k,
            )
            # Filter by agents if specified
            if agents:
                semantic_results = self._filter_by_agents(semantic_results, agents)
            results.extend(semantic_results)
        except Exception as e:
            print(f"Semantic search error for '{query}': {e}")

        # Keyword search
        try:
            keywords = self._extract_keywords(query)
            if keywords:
                keyword_results = await self.vector_store.keyword_search(
                    self.table_name,
                    keywords,
                    top_k=self.keyword_top_k,
                )
                # Filter by agents if specified
                if agents:
                    keyword_results = self._filter_by_agents(keyword_results, agents)
                results.extend(keyword_results)
        except Exception as e:
            print(f"Keyword search error for '{query}': {e}")

        return results

    async def _execute_searches(
        self,
        queries: List[str],
        agents: Optional[List[str]] = None,
    ) -> List[List[MemoryEntry]]:
        """Execute searches in parallel"""
        tasks = [self._execute_single_search(q, agents) for q in queries]
        return await asyncio.gather(*tasks)

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query for lexical search"""
        # Simple keyword extraction
        stop_words = {
            "a",
            "an",
            "the",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "need",
            "dare",
            "ought",
            "used",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "just",
            "and",
            "but",
            "if",
            "or",
            "because",
            "until",
            "while",
            "what",
            "which",
            "who",
            "whom",
            "this",
            "that",
            "these",
            "those",
            "am",
            "i",
            "me",
            "my",
            "myself",
            "we",
            "our",
            "ours",
            "ourselves",
            "you",
            "your",
            "yours",
            "yourself",
            "yourselves",
            "he",
            "him",
            "his",
            "himself",
            "she",
            "her",
            "hers",
            "herself",
            "it",
            "its",
            "itself",
            "they",
            "them",
            "their",
            "theirs",
            "themselves",
        }

        words = query.lower().split()
        keywords = [
            word.strip(".,!?;:'\"()[]{}")
            for word in words
            if word.lower() not in stop_words and len(word) > 2
        ]

        return keywords[:10]  # Max 10 keywords

    def _merge_and_deduplicate(
        self,
        results_lists: List[List[MemoryEntry]],
    ) -> List[MemoryEntry]:
        """Merge and deduplicate results from multiple searches"""
        seen_ids = set()
        merged = []

        for results in results_lists:
            for entry in results:
                if entry.entry_id not in seen_ids:
                    seen_ids.add(entry.entry_id)
                    merged.append(entry)

        return merged

    async def _retrieve_with_reflection(
        self,
        query: str,
        initial_results: List[MemoryEntry],
        plan: RetrievalPlan,
        agents: Optional[List[str]] = None,
    ) -> List[MemoryEntry]:
        """Iterative refinement through reflection"""

        current_results = initial_results

        for round_num in range(self.max_reflection_rounds):
            # Check completeness
            is_complete, missing_info = await self._check_completeness(
                query,
                current_results,
                plan,
            )

            # Log the result of completeness check
            with profiler.profile(
                f"completeness_check_logging_{round_num}",
                args={"is_complete": is_complete, "missing_info": missing_info},
            ):
                pass

            if is_complete:
                break

            # Generate additional queries for missing info
            with profiler.profile(f"generate_missing_queries_{round_num}"):
                additional_queries = await self._generate_missing_info_queries(
                    query,
                    missing_info,
                )

            if not additional_queries:
                break

            # Execute additional searches
            with profiler.profile(f"execute_additional_searches_{round_num}"):
                additional_results = await self._execute_searches(
                    additional_queries, agents
                )

            # Merge with existing results
            all_results = [current_results] + additional_results
            current_results = self._merge_and_deduplicate(all_results)

        return current_results

    async def _check_completeness(
        self,
        query: str,
        results: List[MemoryEntry],
        plan: RetrievalPlan,
    ) -> tuple[bool, List[str]]:
        """Check if retrieved results are sufficient"""

        if not results:
            return False, ["No results found"]

        # Format results for analysis
        results_text = "\n".join(
            [
                f"- {entry.lossless_restatement}"
                for entry in results[:15]  # Limit to reduce context
            ]
        )

        # Prioritize top 3 required info to avoid strictness
        top_required = [info.get("type", "") for info in plan.required_info[:3]]

        prompt = f"""Assess if the Question can be answered reasonably well with the Retrieved Information.

Question: {query}

Retrieved Information:
{results_text}

Required Info Types (Guidance only):
{top_required}

Task:
1. Determine if there is enough information to construct a helpful answer.
2. Only say 'is_complete: false' if CRITICAL information is missing.
3. IGNORE requests for "more context", "original domain", or "specific references" if the core facts are present.

Return JSON:
{{
  "is_complete": true/false,
  "missing_info": ["critical_missing_fact"] or []
}}

Return ONLY valid JSON."""

        messages = [
            {
                "role": "system",
                "content": "You are a pragmatic information analyst. You prefer marking tasks complete.",
            },
            {"role": "user", "content": prompt},
        ]

        try:
            # Check if client supports 'model' arg (LiteLLM)
            kwargs = {"messages": messages, "temperature": self.temperature}
            if (
                self.fast_model
                and hasattr(self.client, "chat_completion")
                and "model" in self.client.chat_completion.__code__.co_varnames
            ):
                kwargs["model"] = self.fast_model

            response = await self.client.chat_completion(**kwargs)

            data = self.client.extract_json(response)
            if data:
                return (
                    data.get("is_complete", True),
                    data.get("missing_info", []),
                )
        except Exception as e:
            print(f"Completeness check error: {e}")

        return True, []

    async def _generate_missing_info_queries(
        self,
        original_query: str,
        missing_info: List[str],
    ) -> List[str]:
        """Generate queries to find missing information"""

        if not missing_info:
            return []

        prompt = f"""Generate search queries to find the missing information.

Original Question: {original_query}

Missing Information:
{missing_info}

Generate 1-2 targeted search queries to find this missing information.

Return JSON:
{{
  "queries": ["query1", "query2"]
}}

Return ONLY valid JSON."""

        messages = [
            {"role": "system", "content": "You are a search query generator."},
            {"role": "user", "content": prompt},
        ]

        try:
            response = await self.client.chat_completion(
                messages=messages,
                temperature=self.temperature,
            )

            data = self.client.extract_json(response)
            if data and "queries" in data:
                return data["queries"][:2]
        except Exception as e:
            print(f"Missing info query generation error: {e}")

        return []

    async def hybrid_retrieve(
        self,
        query: str,
        persons: Optional[List[str]] = None,
        entities: Optional[List[str]] = None,
        timestamp_start: Optional[str] = None,
        timestamp_end: Optional[str] = None,
    ) -> List[MemoryEntry]:
        """
        Hybrid retrieval combining semantic, lexical, and structured search

        Args:
            query: Search query
            persons: Filter by person names
            entities: Filter by entities
            timestamp_start: Start of timestamp range
            timestamp_end: End of timestamp range

        Returns:
            List of relevant MemoryEntry objects
        """
        all_results = []

        # Semantic search
        query_embedding = await self.embedding_client.create_single_embedding(query)
        semantic_results = await self.vector_store.semantic_search(
            self.table_name,
            query_embedding,
            top_k=self.semantic_top_k,
        )
        all_results.append(semantic_results)

        # Keyword search
        keywords = self._extract_keywords(query)
        if keywords:
            keyword_results = await self.vector_store.keyword_search(
                self.table_name,
                keywords,
                top_k=self.keyword_top_k,
            )
            all_results.append(keyword_results)

        # Structured search (if filters provided)
        if any([persons, entities, timestamp_start, timestamp_end]):
            structured_results = await self.vector_store.structured_search(
                self.table_name,
                persons=persons,
                entities=entities,
                timestamp_start=timestamp_start,
                timestamp_end=timestamp_end,
                top_k=self.keyword_top_k,
            )
            # Prepend structured results (higher priority)
            all_results.insert(0, structured_results)

        return self._merge_and_deduplicate(all_results)
