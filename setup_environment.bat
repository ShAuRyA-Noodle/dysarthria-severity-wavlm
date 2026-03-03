@echo off
REM ============================================================
REM  ATD Project — Environment Setup Script (Windows)
REM  Creates conda env "dysarthria" with Python 3.11,
REM  installs PyTorch 2.2.2 + CUDA 12.1, and all pip deps.
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

REM Step 3: Install PyTorch with CUDA 12.1
echo [3/4] Installing PyTorch 2.2.2 with CUDA 12.1 support...
call conda install pytorch==2.2.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to install PyTorch.
    exit /b 1
)

REM Step 4: Install remaining packages from requirements.txt
echo [4/4] Installing remaining packages from requirements.txt...
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to install pip packages.
    exit /b 1
)

echo.
echo ============================================================
echo  Environment setup complete
echo ============================================================
