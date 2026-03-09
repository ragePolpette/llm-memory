@echo off
REM Script per fermare il server MCP in esecuzione

echo Ricerca processi Python in esecuzione...

REM Cerca processi python che eseguono mcp_server
for /f "tokens=2" %%i in ('tasklist /FI "IMAGENAME eq python.exe" /FO LIST ^| findstr /C:"PID:"') do (
    wmic process where "ProcessId=%%i" get CommandLine | findstr "mcp_server" >nul
    if not errorlevel 1 (
        echo Terminazione processo %%i...
        taskkill /PID %%i /F
    )
)

echo.
echo Server MCP fermato.
pause
