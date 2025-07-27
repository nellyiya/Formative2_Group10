@echo off
echo ================================
echo  MULTIMODAL AUTH SYSTEM DEMO
echo ================================
echo.
echo Starting the demonstration...
echo.

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Run the simple demo
echo Running authentication system demo...
python simple_demo.py

echo.
echo Demo completed!
pause
