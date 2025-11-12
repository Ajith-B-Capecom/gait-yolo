@echo off
echo ============================================================
echo   OpenGait Project - Automated Setup
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/
    pause
    exit /b 1
)

echo Step 1: Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)
echo   [OK] Virtual environment created
echo.

echo Step 2: Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo   [OK] Virtual environment activated
echo.

echo Step 3: Upgrading pip...
python -m pip install --upgrade pip
echo   [OK] Pip upgraded
echo.

echo Step 4: Installing required packages...
echo This may take several minutes...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install packages
    pause
    exit /b 1
)
echo   [OK] All packages installed
echo.

echo Step 5: Creating folder structure...
python setup.py
echo   [OK] Folders created
echo.

echo ============================================================
echo   Setup Complete!
echo ============================================================
echo.
echo Next Steps:
echo 1. Add video files to: data\videos\person1\
echo 2. Add video files to: data\videos\person2\
echo 3. Add video files to: data\videos\person3\
echo 4. Run: python main.py
echo.
echo To activate virtual environment later, run:
echo   venv\Scripts\activate.bat
echo.
pause
