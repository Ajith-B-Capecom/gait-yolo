# OpenGait Project - Automated Setup (PowerShell)

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  OpenGait Project - Automated Setup" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python detected: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python from https://www.python.org/" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "Step 1: Creating virtual environment..." -ForegroundColor Yellow
python -m venv venv
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to create virtual environment" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "  [OK] Virtual environment created" -ForegroundColor Green
Write-Host ""

Write-Host "Step 2: Activating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to activate virtual environment" -ForegroundColor Red
    Write-Host "You may need to run: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "  [OK] Virtual environment activated" -ForegroundColor Green
Write-Host ""

Write-Host "Step 3: Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip | Out-Null
Write-Host "  [OK] Pip upgraded" -ForegroundColor Green
Write-Host ""

Write-Host "Step 4: Installing required packages..." -ForegroundColor Yellow
Write-Host "This may take several minutes..." -ForegroundColor Cyan
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install packages" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "  [OK] All packages installed" -ForegroundColor Green
Write-Host ""

Write-Host "Step 5: Creating folder structure..." -ForegroundColor Yellow
python setup.py
Write-Host "  [OK] Folders created" -ForegroundColor Green
Write-Host ""

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "1. Add video files to: data\videos\person1\" -ForegroundColor White
Write-Host "2. Add video files to: data\videos\person2\" -ForegroundColor White
Write-Host "3. Add video files to: data\videos\person3\" -ForegroundColor White
Write-Host "4. Run: python main.py" -ForegroundColor White
Write-Host ""
Write-Host "To activate virtual environment later, run:" -ForegroundColor Yellow
Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Read-Host "Press Enter to exit"
