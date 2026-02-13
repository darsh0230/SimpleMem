"""
LiteLLM-compatible client using OpenAI SDK for vLLM/OpenAI-format servers.

Uses AsyncOpenAI with custom base_url to connect to hosted models via vLLM.
Supports optional proxy configuration for production environments.
"""

import json
import re
import hashlib
from typing import List, Dict, Any, Optional, AsyncIterator

import httpx
from openai import AsyncOpenAI


class LiteLLMClient:
    """
    LiteLLM-compatible client for vLLM/OpenAI-format servers.
    Uses AsyncOpenAI SDK with custom base_url for proxy support.
    """

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        llm_model: str = "open-large",
        embedding_model: str = "text-embedding-3-small",
        proxy_url: Optional[str] = None,
        timeout: float = 120.0,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.timeout = timeout

        # Configure HTTP client with optional proxy
        client_kwargs: Dict[str, Any] = {
            "api_key": api_key,
            "timeout": timeout,
        }

        if base_url:
            client_kwargs["base_url"] = base_url

        if proxy_url:
            try:
                # Check if proxy_url is a JSON string (for OUTGOING_HTTP_PROXY format)
                if proxy_url.strip().startswith("{"):
                    try:
                        import json

                        proxy_config = json.loads(proxy_url)
                        print(
                            f"DEBUG: Parsed proxy config: {proxy_config}"
                        )  # Debug logging

                        mounts = {}
                        # Handle http:// and https:// prefixes or keys without protocol
                        http_proxy = proxy_config.get("http://") or proxy_config.get(
                            "http"
                        )
                        https_proxy = proxy_config.get("https://") or proxy_config.get(
                            "https"
                        )

                        if http_proxy:
                            mounts["http://"] = httpx.AsyncHTTPTransport(
                                proxy=http_proxy
                            )

                        if https_proxy:
                            mounts["https://"] = httpx.AsyncHTTPTransport(
                                proxy=https_proxy
                            )

                        if mounts:
                            http_client = httpx.AsyncClient(
                                mounts=mounts, timeout=timeout
                            )
                        else:
                            # Fallback if JSON parsed but no valid keys found
                            http_client = httpx.AsyncClient(
                                proxy=proxy_url, timeout=timeout
                            )

                    except json.JSONDecodeError:
                        # Fallback to simple string if JSON parse fails
                        http_client = httpx.AsyncClient(
                            proxy=proxy_url, timeout=timeout
                        )
                else:
                    http_client = httpx.AsyncClient(proxy=proxy_url, timeout=timeout)

                client_kwargs["http_client"] = http_client
            except Exception as e:
                print(f"Warning: Could not configure proxy: {e}")

        self.client = AsyncOpenAI(**client_kwargs)

    async def close(self):
        """Close the HTTP client"""
        if hasattr(self.client, "_client") and self.client._client:
            await self.client._client.aclose()

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict] = None,
        stream: bool = False,
    ) -> str:
        """
        Call LLM for chat completion.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
            response_format: Optional response format (e.g., {"type": "json_object"})
            stream: Whether to stream the response (ignored, use chat_completion_stream)

        Returns:
            Generated text content
        """
        request_params: Dict[str, Any] = {
            "model": self.llm_model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }

        if max_tokens:
            request_params["max_tokens"] = max_tokens

        if response_format:
            request_params["response_format"] = response_format

        response = await self.client.chat.completions.create(**request_params)
        return response.choices[0].message.content or ""

    async def chat_completion_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict] = None,
    ) -> AsyncIterator[str]:
        """
        Stream chat completion chunks.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
            response_format: Optional response format

        Yields:
            Text content chunks
        """
        request_params: Dict[str, Any] = {
            "model": self.llm_model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }

        if max_tokens:
            request_params["max_tokens"] = max_tokens

        if response_format:
            request_params["response_format"] = response_format

        stream = await self.client.chat.completions.create(**request_params)

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def create_embedding(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        """
        Create embeddings for texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        embeddings = []
        for text in texts:
            response = await self.client.embeddings.create(
                model=self.embedding_model,
                input=text,
            )
            embeddings.append(response.data[0].embedding)
        return embeddings

    async def create_single_embedding(self, text: str) -> List[float]:
        """Create embedding for a single text"""
        embeddings = await self.create_embedding([text])
        return embeddings[0]

    async def verify_api_key(
        self, model: Optional[str] = None
    ) -> tuple[bool, Optional[str]]:
        """
        Verify that the API key and connection work.

        Args:
            model: Optional model to test (defaults to llm_model)

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            test_model = model or self.llm_model
            # Try a minimal completion to verify
            await self.client.chat.completions.create(
                model=test_model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
            )
            return True, None
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "unauthorized" in error_msg.lower():
                return False, "Invalid API key"
            elif "404" in error_msg:
                return False, f"Model not found: {model or self.llm_model}"
            else:
                return False, f"Connection error: {error_msg}"

    def extract_json(self, text: str) -> Any:
        """
        Extract JSON from LLM response text with robust parsing.

        Args:
            text: Raw text that may contain JSON

        Returns:
            Parsed JSON object or None
        """
        if not text:
            return None

        # Strategy 1: Direct JSON parsing
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Strategy 2: Extract from ```json ``` blocks
        json_block_pattern = r"```json\s*([\s\S]*?)\s*```"
        matches = re.findall(json_block_pattern, text, re.IGNORECASE)
        if matches:
            try:
                return json.loads(matches[0].strip())
            except json.JSONDecodeError:
                pass

        # Strategy 3: Extract from generic ``` ``` blocks
        generic_block_pattern = r"```\s*([\s\S]*?)\s*```"
        matches = re.findall(generic_block_pattern, text)
        if matches:
            for match in matches:
                try:
                    return json.loads(match.strip())
                except json.JSONDecodeError:
                    continue

        # Strategy 4: Find balanced JSON object/array
        start_obj = text.find("{")
        start_arr = text.find("[")

        if start_obj == -1 and start_arr == -1:
            return None

        if start_arr == -1 or (start_obj != -1 and start_obj < start_arr):
            json_str = self._extract_balanced_braces(text[start_obj:], "{", "}")
        else:
            json_str = self._extract_balanced_braces(text[start_arr:], "[", "]")

        if json_str:
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

        # Strategy 5: Clean and retry
        cleaned = self._clean_json_string(text)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        return None

    def _extract_balanced_braces(
        self, text: str, open_char: str, close_char: str
    ) -> Optional[str]:
        """Extract a balanced brace-enclosed string"""
        if not text or text[0] != open_char:
            return None

        count = 0
        in_string = False
        escape_next = False

        for i, char in enumerate(text):
            if escape_next:
                escape_next = False
                continue

            if char == "\\":
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if in_string:
                continue

            if char == open_char:
                count += 1
            elif char == close_char:
                count -= 1
                if count == 0:
                    return text[: i + 1]

        return None

    def _clean_json_string(self, text: str) -> str:
        """Clean common JSON issues from LLM output"""
        prefixes = [
            "Here's the JSON:",
            "Here is the JSON:",
            "JSON output:",
            "Output:",
            "Result:",
        ]
        cleaned = text.strip()
        for prefix in prefixes:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix) :].strip()

        # Remove trailing commas before } or ]
        cleaned = re.sub(r",\s*([\}\]])", r"\1", cleaned)

        # Remove single-line comments
        cleaned = re.sub(r"//.*$", "", cleaned, flags=re.MULTILINE)

        return cleaned


class LiteLLMClientManager:
    """
    Manages LiteLLMClient instances.
    Creates and caches clients based on API key hash.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        llm_model: str = "open-large",
        embedding_model: str = "text-embedding-3-small",
        proxy_url: Optional[str] = None,
    ):
        self.base_url = base_url
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.proxy_url = proxy_url
        self._clients: Dict[str, LiteLLMClient] = {}

    def get_client(self, api_key: str) -> LiteLLMClient:
        """
        Get or create a LiteLLMClient for the given API key.

        Args:
            api_key: API key for the client

        Returns:
            LiteLLMClient instance
        """
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]

        if key_hash not in self._clients:
            self._clients[key_hash] = LiteLLMClient(
                api_key=api_key,
                base_url=self.base_url,
                llm_model=self.llm_model,
                embedding_model=self.embedding_model,
                proxy_url=self.proxy_url,
            )

        return self._clients[key_hash]

    async def close_all(self):
        """Close all client connections"""
        for key_hash, client in list(self._clients.items()):
            try:
                await client.close()
            except Exception as e:
                print(f"Warning: Failed to close client {key_hash}: {e}")
        self._clients.clear()

    async def remove_client(self, api_key: str):
        """Remove and close a specific client"""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
        if key_hash in self._clients:
            try:
                await self._clients[key_hash].close()
            except Exception as e:
                print(f"Warning: Failed to close client {key_hash}: {e}")
            finally:
                del self._clients[key_hash]
