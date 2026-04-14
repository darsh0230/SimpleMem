"""
SimpleMem MCP Server Package

A multi-tenant memory service for LLM agents via the MCP (Model Context Protocol).

This package can be used standalone or mounted in an existing FastAPI application.
"""

from .app_factory import create_app

__version__ = "1.0.7"
__all__ = ["create_app"]
