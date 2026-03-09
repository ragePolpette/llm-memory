@echo off
REM Script di avvio silenzioso per LLM Memory MCP HTTP Server
REM Ideale per autostart - avvia in background senza finestra

REM Cambia directory al progetto
cd /d "%~dp0"

REM Default runtime v2 locale-only
if "%EMBEDDING_PROVIDER%"=="" set EMBEDDING_PROVIDER=hash-local
if "%EMBEDDING_MODEL%"=="" set EMBEDDING_MODEL=local-hash-v1
if "%MEMORY_STORAGE_BACKEND%"=="" set MEMORY_STORAGE_BACKEND=sqlite
if "%MEMORY_VECTOR_BACKEND%"=="" set MEMORY_VECTOR_BACKEND=sqlite
if "%MEMORY_SQLITE_PATH%"=="" set MEMORY_SQLITE_PATH=./data/memory.db
if "%MEMORY_ALLOW_OUTBOUND_NETWORK%"=="" set MEMORY_ALLOW_OUTBOUND_NETWORK=false

REM Crea directory per i log se non esiste
if not exist "logs" mkdir logs

REM Genera nome file log con timestamp
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set LOG_FILE=logs\mcp_server_%datetime:~0,8%_%datetime:~8,6%.log

REM Avvia il server HTTP in finestra minimizzata con output su log
start /MIN python -m src.mcp_server.http_server > %LOG_FILE% 2>&1

REM Crea file PID per tracking
echo %date% %time% > logs\last_start.txt

exit
