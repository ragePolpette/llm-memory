@echo off
REM Wrapper per avviare llm-memory MCP HTTP server da qualsiasi directory
cd /d "%~dp0\.."
if "%EMBEDDING_PROVIDER%"=="" set EMBEDDING_PROVIDER=hash-local
if "%EMBEDDING_MODEL%"=="" set EMBEDDING_MODEL=local-hash-v1
if "%MEMORY_STORAGE_BACKEND%"=="" set MEMORY_STORAGE_BACKEND=sqlite
if "%MEMORY_VECTOR_BACKEND%"=="" set MEMORY_VECTOR_BACKEND=sqlite
if "%MEMORY_SQLITE_PATH%"=="" set MEMORY_SQLITE_PATH=./data/memory.db
if "%MEMORY_ALLOW_OUTBOUND_NETWORK%"=="" set MEMORY_ALLOW_OUTBOUND_NETWORK=false
python -m src.mcp_server.http_server
