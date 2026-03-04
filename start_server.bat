@echo off
REM Script di avvio per LLM Memory MCP Server
REM Può essere aggiunto all'autostart di Windows

echo ========================================
echo   LLM Memory MCP HTTP Server
echo ========================================
echo.

REM Cambia directory al progetto
cd /d "%~dp0"

REM Default runtime v2 locale-only
if "%EMBEDDING_PROVIDER%"=="" set EMBEDDING_PROVIDER=hash-local
if "%EMBEDDING_MODEL%"=="" set EMBEDDING_MODEL=local-hash-v1
if "%MEMORY_STORAGE_BACKEND%"=="" set MEMORY_STORAGE_BACKEND=sqlite
if "%MEMORY_VECTOR_BACKEND%"=="" set MEMORY_VECTOR_BACKEND=sqlite
if "%MEMORY_SQLITE_PATH%"=="" set MEMORY_SQLITE_PATH=./data/memory.db
if "%MEMORY_ALLOW_OUTBOUND_NETWORK%"=="" set MEMORY_ALLOW_OUTBOUND_NETWORK=false

REM Verifica che Python sia disponibile
python --version >nul 2>&1
if errorlevel 1 (
    echo ERRORE: Python non trovato nel PATH
    echo Installa Python o aggiungilo al PATH
    pause
    exit /b 1
)

REM Crea directory per i log se non esiste
if not exist "logs" mkdir logs

REM Genera nome file log con timestamp
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set LOG_FILE=logs\mcp_server_%datetime:~0,8%_%datetime:~8,6%.log

echo Avvio server MCP HTTP...
echo Log file: %LOG_FILE%
echo Endpoint: http://127.0.0.1:8767/mcp
echo.
echo Per fermare il server, chiudi questa finestra o premi Ctrl+C
echo ========================================
echo.

REM Avvia il server con output su file e console (compatibile Windows CMD)
where powershell >nul 2>&1
if errorlevel 1 (
    echo [WARN] powershell non trovato: output solo su log.
    python -m src.mcp_server.http_server > "%LOG_FILE%" 2>&1
) else (
    powershell -NoProfile -Command "python -m src.mcp_server.http_server 2>&1 | Tee-Object -FilePath '%LOG_FILE%'"
)

REM Se il server termina con errore
if errorlevel 1 (
    echo.
    echo ========================================
    echo ERRORE: Il server è terminato con errori
    echo Controlla il log: %LOG_FILE%
    echo ========================================
    pause
)
