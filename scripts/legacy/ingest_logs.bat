@echo off
REM Script per ingestare i Daily Log in llm-memory
REM Uso: ingest_logs.bat [--dry-run] [--force]

cd /d "%~dp0\.."
if "%DECISION_LOG_DIR%"=="" set DECISION_LOG_DIR=%~dp0..\..\Decision_Log
python scripts/ingest_logs.py --source "%DECISION_LOG_DIR%" %*

pause
