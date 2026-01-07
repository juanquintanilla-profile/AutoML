@echo off
REM AutoML Agent - Start script with auto-browser open
REM Usage: start.bat

echo Starting AutoML services...
docker-compose up -d

echo Waiting for services to start...
:WAIT_LOOP
timeout /t 3 /nobreak >nul
curl -s http://localhost:8501 >nul 2>&1
if errorlevel 1 (
    echo Still waiting for Streamlit UI...
    goto WAIT_LOOP
)

echo Services are ready!
echo Opening browser...
start http://localhost:8501

echo.
echo AutoML is running:
echo   - UI: http://localhost:8501
echo   - API: http://localhost:8000/docs
echo.
echo To view logs: docker-compose logs -f
echo To stop: docker-compose down
