# SimpleMem MCP Server - Python Package

A multi-tenant memory service for LLM agents via the MCP (Model Context Protocol). This package can be used standalone or integrated into existing FastAPI applications.

## Installation

### From Git Repository

```bash
pip install git+https://github.com/yourusername/simplemem-mcp.git
```

### Local Development

```bash
# Clone the repository
git clone https://github.com/yourusername/simplemem-mcp.git
cd simplemem-mcp

# Install in editable mode
pip install -e .
```

## Usage

### Standalone Server

Run the server directly:

```bash
# Using the installed command
simplemem-mcp

# Or using Python module
python -m simplemem_mcp.server
```

### Mount in Existing FastAPI Application

```python
from fastapi import FastAPI
from simplemem_mcp import create_app

# Your existing FastAPI app
main_app = FastAPI(title="My Application")

# Create MCP app instance
mcp_app = create_app()

# Mount under a prefix
main_app.mount("/simplemem", mcp_app)

# Now all MCP routes are available under /simplemem prefix:
# - /simplemem/api/auth/register
# - /simplemem/api/auth/verify
# - /simplemem/mcp (MCP protocol endpoint)
# - etc.
```

### Custom Mounting Example

```python
from fastapi import FastAPI
from simplemem_mcp import create_app
import uvicorn

# Create your main application
app = FastAPI(
    title="My AI Application",
    description="Application with integrated memory service"
)

# Add your own routes
@app.get("/")
async def root():
    return {"message": "Welcome to my AI application"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

# Mount SimpleMem MCP under /memory prefix
mcp_app = create_app()
app.mount("/memory", mcp_app)

# Run the combined application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

Now your application has:
- Your routes: `/`, `/health`
- MCP routes: `/memory/api/auth/register`, `/memory/mcp`, etc.

## Configuration

SimpleMem MCP is configured via environment variables. Create a `.env` file or set these in your environment:

### Required Environment Variables

```bash
# JWT Configuration
JWT_SECRET_KEY=your-secret-key-here
ENCRYPTION_KEY=your-32-byte-encryption-key-base64

# Database Paths
USER_DB_PATH=./data/users.db
LANCEDB_PATH=./data/lancedb

# LLM Provider (choose one: openrouter, ollama, litellm)
LLM_PROVIDER=litellm
```

### LiteLLM Provider (Recommended)

```bash
LLM_PROVIDER=litellm
LITELLM_BASE_URL=https://your-litellm-proxy.com
LITELLM_LLM_MODEL=gpt-4
LITELLM_EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536
```

### OpenRouter Provider

```bash
LLM_PROVIDER=openrouter
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=anthropic/claude-3-opus
EMBEDDING_MODEL=openai/text-embedding-3-small
EMBEDDING_DIMENSION=1536
```

### Ollama Provider (Local)

```bash
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=llama2
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIMENSION=768
```

### Optional Configuration

```bash
# Server Configuration
MCP_HOST=0.0.0.0
MCP_PORT=8000
MCP_BASE_URL=https://your-domain.com

# Memory Configuration
WINDOW_SIZE=10
TOP_K=5

# JWT Expiration
JWT_EXPIRATION_DAYS=30
```

## Environment Variable Reference

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `JWT_SECRET_KEY` | Secret key for JWT token generation | - | Yes |
| `ENCRYPTION_KEY` | Base64-encoded 32-byte key for API key encryption | - | Yes |
| `USER_DB_PATH` | Path to SQLite user database | `./data/users.db` | No |
| `LANCEDB_PATH` | Path to LanceDB vector database | `./data/lancedb` | No |
| `LLM_PROVIDER` | LLM provider to use | `litellm` | No |
| `LITELLM_BASE_URL` | LiteLLM proxy URL | - | If using LiteLLM |
| `LITELLM_LLM_MODEL` | LLM model name for LiteLLM | `gpt-4` | No |
| `LITELLM_EMBEDDING_MODEL` | Embedding model for LiteLLM | `text-embedding-3-small` | No |
| `OPENROUTER_BASE_URL` | OpenRouter API URL | `https://openrouter.ai/api/v1` | No |
| `LLM_MODEL` | LLM model name (OpenRouter/Ollama) | `anthropic/claude-3-opus` | No |
| `EMBEDDING_MODEL` | Embedding model (OpenRouter/Ollama) | `openai/text-embedding-3-small` | No |
| `OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` | If using Ollama |
| `EMBEDDING_DIMENSION` | Dimension of embedding vectors | `1536` | No |
| `WINDOW_SIZE` | Context window size for memory retrieval | `10` | No |
| `TOP_K` | Number of top memories to retrieve | `5` | No |
| `JWT_EXPIRATION_DAYS` | JWT token expiration in days | `30` | No |
| `MCP_HOST` | Server host address | `0.0.0.0` | No |
| `MCP_PORT` | Server port | `8000` | No |
| `MCP_BASE_URL` | Base URL for MCP endpoints | - | No |

## API Endpoints

When mounted at `/simplemem`:

### Authentication
- `POST /simplemem/api/auth/register` - Register new user
- `GET /simplemem/api/auth/verify` - Verify token
- `POST /simplemem/api/auth/refresh` - Refresh token

### Health & Info
- `GET /simplemem/api/health` - Health check
- `GET /simplemem/api/server/info` - Server information

### MCP Protocol
- `POST /simplemem/mcp` - MCP message endpoint (Streamable HTTP)
- `GET /simplemem/mcp` - MCP SSE stream (server-to-client)
- `DELETE /simplemem/mcp` - Terminate MCP session
- `GET /simplemem/mcp/sse` - Legacy SSE endpoint

## Development

### Running Tests

```bash
pip install -e ".[dev]"
pytest
```

### Project Structure

```
simplemem-mcp/
├── server/
│   ├── __init__.py          # Package exports
│   ├── __main__.py          # CLI entry point
│   ├── app_factory.py       # FastAPI app factory
│   ├── http_server.py       # Main server implementation
│   ├── mcp_handler.py       # MCP protocol handler
│   ├── auth/                # Authentication modules
│   ├── core/                # Core memory logic
│   ├── database/            # Database interfaces
│   ├── integrations/        # LLM provider integrations
│   └── utils/               # Utility functions
├── config/
│   └── settings.py          # Configuration management
├── setup.py                 # Package setup
├── pyproject.toml           # Modern Python packaging
└── README_PACKAGE.md        # This file
```

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions, please open an issue on GitHub.
