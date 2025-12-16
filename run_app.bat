@echo off
Title UMLGenie Launcher
echo ====================================================
echo          Starting UMLGenie Smart Assistant
echo ====================================================
echo.

streamlit run app.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Application exited with error code %errorlevel%.
    echo Please check if the environment is active and dependencies are installed.
    pause
)
