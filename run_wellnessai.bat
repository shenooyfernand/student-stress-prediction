@echo off
echo ========================================
echo    WellnessAI - Student Stress Monitor
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed!
    echo.
    echo Please download Python from: https://www.python.org/downloads/
    echo Make sure to CHECK "Add Python to PATH" during installation
    echo.
    pause
    exit /b
)

echo ✅ Python is installed
echo.

REM Install packages
echo Step 1: Installing required packages...
pip install pandas numpy scikit-learn joblib flask

echo.
echo Step 2: Creating ML model...
python create_model.py

echo.
echo Step 3: Starting WellnessAI...
echo.
echo ========================================
echo    OPEN YOUR BROWSER AND GO TO:
echo    http://localhost:5000
echo ========================================
echo.
python app.py

pause