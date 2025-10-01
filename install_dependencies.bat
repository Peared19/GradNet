@echo off
echo Installing GradNet dependencies...
echo.

REM Check if pip is available
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: pip is not installed or not in PATH
    echo Please install Python with pip first
    pause
    exit /b 1
)

echo Installing packages from requirements.txt...
pip install -r requirements.txt

if %errorlevel% equ 0 (
    echo.
    echo ✅ All dependencies installed successfully!
    echo.
    echo You can now run:
    echo   - python get_images_coco.py     (to download images)
    echo   - python preprocess_sobel.py    (to generate edge maps)
    echo   - python train_edge_detector.py (to train the model)
    echo   - python test.py                (to test the model)
) else (
    echo.
    echo ❌ Error occurred during installation
    echo Please check the error messages above
)

echo.
pause
