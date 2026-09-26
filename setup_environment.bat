@echo off
REM ============================================================
REM  ATD Project — Environment Setup Script (Windows)
REM  Creates conda env "dysarthria" with Python 3.11,
REM  installs matching CUDA 12.6 PyTorch wheels and project requirements.
REM ============================================================

echo ============================================================
echo  ATD Project — Environment Setup
echo ============================================================

REM Step 1: Create conda environment
echo [1/4] Creating conda environment "dysarthria" with Python 3.11...
call conda create -n dysarthria python=3.11 -y
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to create conda environment.
    exit /b 1
)

REM Step 2: Activate environment
echo [2/4] Activating "dysarthria" environment...
call conda activate dysarthria
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to activate conda environment.
    exit /b 1
)

REM Step 3: Install matching GPU-enabled PyTorch packages
echo [3/4] Installing PyTorch 2.13.0 and Torchaudio 2.11.0 with CUDA 12.6...
python -m pip install torch==2.13.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu126
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to install CUDA-enabled PyTorch packages.
    exit /b 1
)

REM Step 4: Install the remaining compatible dependencies
echo [4/4] Installing packages from requirements.txt...
python -m pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to install pip packages.
    exit /b 1
)

echo.
echo ============================================================
echo  Environment setup complete
echo ============================================================
