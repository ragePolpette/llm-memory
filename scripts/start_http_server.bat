@echo off
REM Avvia LLM Memory HTTP Server per connessione manuale
REM Avvia PRIMA di aprire VS Code/Antigravity

echo ============================================
echo   LLM Memory HTTP Server
echo   Avvia questo PRIMA di Antigravity
echo ============================================
echo.

cd /d "%~dp0\.."

REM Setta variabili per silenziare warning HF
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set HF_HUB_OFFLINE=1

echo Avvio server su http://127.0.0.1:8767 ...
echo (il modello impiega ~15-30 sec a caricarsi)
echo.

python -m src.mcp_server.http_server

pause
