@echo off
REM Quick start script for Wikipedia training on Windows
REM Works in PyCharmProjectsSpaceConflict folder
REM Usage: scripts\quick_start_wikipedia.bat [cpu|gpu|both|test|fast-test]

setlocal enabledelayedexpansion

REM Get script directory and project root
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..

echo ============================================
echo DeepSeek-V3 Wikipedia Training Quick Start
echo ============================================
echo Project root: %PROJECT_ROOT%
echo.

REM Change to project root
cd /d "%PROJECT_ROOT%"

set MODE=%1
if "%MODE%"=="" set MODE=both

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed
    exit /b 1
)

echo Python version:
python --version
echo.

REM Check dependencies
echo Checking dependencies...
python -c "import torch; import transformers; import datasets" 2>nul
if errorlevel 1 (
    echo Warning: Some dependencies are missing. Installing...
    pip install -r requirements.txt
)

echo Dependencies OK
echo.

REM Create necessary directories
echo Creating directories...
if not exist wikipedia_checkpoints\cpu mkdir wikipedia_checkpoints\cpu
if not exist wikipedia_checkpoints\gpu mkdir wikipedia_checkpoints\gpu
if not exist wikipedia_cache\sanitized mkdir wikipedia_cache\sanitized
if not exist logs\cpu_wikipedia mkdir logs\cpu_wikipedia
if not exist logs\gpu_wikipedia mkdir logs\gpu_wikipedia
echo Directories created
echo.

if "%MODE%"=="cpu" goto train_cpu
if "%MODE%"=="gpu" goto train_gpu
if "%MODE%"=="both" goto train_both
if "%MODE%"=="test" goto run_test
if "%MODE%"=="quick-test" goto quick_test
if "%MODE%"=="fast-test" goto fast_test
if "%MODE%"=="hiroshima" goto test_hiroshima
goto usage

:train_cpu
echo ==========================================
echo Starting CPU Training
echo ==========================================
python scripts\train_wikipedia_unified.py --config configs\deepseek_v3_cpu_wikipedia.json --device cpu
goto end

:train_gpu
echo ==========================================
echo Starting GPU Training
echo ==========================================
python scripts\train_wikipedia_unified.py --config configs\deepseek_v3_gpu_wikipedia.json --device cuda
goto end

:train_both
echo ==========================================
echo Training Both CPU and GPU Models
echo ==========================================
echo.
echo Starting CPU training first...
python scripts\train_wikipedia_unified.py --config configs\deepseek_v3_cpu_wikipedia.json --device cpu

echo.
echo CPU training complete. Starting GPU training...
python scripts\train_wikipedia_unified.py --config configs\deepseek_v3_gpu_wikipedia.json --device cuda
goto end

:run_test
echo ==========================================
echo Running Test Suite
echo ==========================================
python scripts\test_wikipedia_training.py
goto end

:quick_test
echo ==========================================
echo Running Quick Test (100 steps)
echo ==========================================
python scripts\test_wikipedia_training.py --quick --steps 100
goto end

:test_hiroshima
echo ==========================================
echo Testing Hiroshima Prompt
echo ==========================================
python scripts\test_hiroshima_prompt.py --test-both
goto end

:fast_test
echo ==========================================
echo Running Fast 500K Tests (CPU 12min / GPU 75s)
echo ==========================================
python tests\test_fast_500k_training.py
goto end

:usage
echo Usage: %0 [cpu^|gpu^|both^|test^|quick-test^|fast-test^|hiroshima]
echo.
echo Options:
echo   cpu         - Train on CPU only
echo   gpu         - Train on GPU only
echo   both        - Train both CPU and GPU sequentially
echo   test        - Run full test suite
echo   quick-test  - Run quick test (100 steps)
echo   fast-test   - Run fast 500K parameter tests (12min CPU / 75s GPU)
echo   hiroshima   - Test Hiroshima prompt on trained models
exit /b 1

:end
echo.
echo ============================================
echo Complete!
echo ============================================