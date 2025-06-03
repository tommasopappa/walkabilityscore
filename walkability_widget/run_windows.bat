@echo off
echo Starting Walkability Explorer...
echo.

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo No virtual environment found. Creating one...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo Installing requirements...
    pip install -r requirements.txt
)

echo.
echo Launching Streamlit application...
echo.
echo The application will open in your default web browser.
echo Press Ctrl+C in this window to stop the server.
echo.

streamlit run app_with_routing.py

pause
