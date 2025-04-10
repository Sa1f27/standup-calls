@echo off
setlocal

echo ===== Starting FastAPI Server ===== >> "C:\Users\Administrator\Desktop\PROJECTS\standup-calls\server_log.txt"
cd /d "C:\Users\Administrator\Desktop\PROJECTS\standup-calls"

REM Activate virtual environment
call stcvenv\Scripts\activate.bat >> server_log.txt 2>&1

REM Wait for SQL Server to start (adjust time as needed)
timeout /t 30 /nobreak >nul

REM Start FastAPI server
start "" cmd /c "uvicorn main:app --host 192.168.48.200 --port 443 --ssl-keyfile=key.pem --ssl-certfile=cert.pem >> server_log.txt 2>&1"

timeout /t 3 /nobreak >nul

REM Launch Chrome with bypass flags
start chrome https://192.168.48.200 --ignore-certificate-errors --user-data-dir="C:\Temp\ChromeTest"

endlocal
exit
