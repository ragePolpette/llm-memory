@echo off
REM Script di avvio per LLM Memory MCP Server
REM Può essere aggiunto all'autostart di Windows

echo ========================================
echo   LLM Memory MCP Server
echo ========================================
echo.

REM Cambia directory al progetto
cd /d "%~dp0"

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

echo Avvio server MCP...
echo Log file: %LOG_FILE%
echo.
echo Per fermare il server, chiudi questa finestra o premi Ctrl+C
echo ========================================
echo.

REM Avvia il server con output su file e console
python -m src.mcp_server.server 2>&1 | tee %LOG_FILE%

REM Se il server termina con errore
if errorlevel 1 (
    echo.
    echo ========================================
    echo ERRORE: Il server è terminato con errori
    echo Controlla il log: %LOG_FILE%
    echo ========================================
    pause
)
