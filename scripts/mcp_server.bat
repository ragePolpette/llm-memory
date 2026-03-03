@echo off
REM Wrapper per avviare llm-memory MCP server da qualsiasi directory
cd /d "%~dp0\.."
python -m src.mcp_server.server
