@echo off
REM Avvia LLM Memory HTTP Server per connessione manuale
REM Avvia PRIMA di aprire VS Code/Antigravity

echo ============================================
echo   LLM Memory HTTP Server
echo   Avvia questo PRIMA di Antigravity
echo ============================================
echo.

cd /d "%~dp0\.."

REM Runtime locale-only hard-safe
if "%EMBEDDING_PROVIDER%"=="" set EMBEDDING_PROVIDER=hash-local
if "%EMBEDDING_MODEL%"=="" set EMBEDDING_MODEL=local-hash-v1
if "%MEMORY_STORAGE_BACKEND%"=="" set MEMORY_STORAGE_BACKEND=sqlite
if "%MEMORY_VECTOR_BACKEND%"=="" set MEMORY_VECTOR_BACKEND=sqlite
if "%MEMORY_SQLITE_PATH%"=="" set MEMORY_SQLITE_PATH=./data/memory.db
if "%MEMORY_ALLOW_OUTBOUND_NETWORK%"=="" set MEMORY_ALLOW_OUTBOUND_NETWORK=false

REM Silenzia warning/telemetry HuggingFace
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set HF_HUB_OFFLINE=1
set HF_HUB_DISABLE_TELEMETRY=1
set TRANSFORMERS_OFFLINE=1
set DO_NOT_TRACK=1
set ANONYMIZED_TELEMETRY=False

echo Avvio server su http://127.0.0.1:8767 ...
echo Provider embedding: %EMBEDDING_PROVIDER% (%EMBEDDING_MODEL%)
echo SQLite: %MEMORY_SQLITE_PATH%
echo.

python -m src.mcp_server.http_server

pause
