@echo off
REM DELPHIC-LLM: Quick test runner for Windows (~$3-5, 15-20 minutes)
REM 3 projects, 1 seed, 2 conditions (DELPHIC-LLM full + Single LLM baseline)

echo.
echo ============================================================
echo  DELPHIC-LLM Quick Test
echo  3 projects -- ~$3-5 USD -- ~15-20 minutes
echo ============================================================
echo.

if "%OPENAI_API_KEY%"=="" (
    echo ERROR: OPENAI_API_KEY not set.
    echo.
    echo Run this first:
    echo   set OPENAI_API_KEY=sk-...
    echo.
    pause
    exit /b 1
)

if not exist "data\nasa93.csv" (
    echo ERROR: data\nasa93.csv not found.
    echo.
    echo Copy nasa93.csv into the data\ subfolder:
    echo   mkdir data
    echo   copy nasa93.csv data\
    echo.
    pause
    exit /b 1
)

if not exist "results\test" mkdir results\test

echo [1/3] Installing Python dependencies (skip if already installed)...
pip install openai pydantic sentence-transformers numpy pandas scipy scikit-learn -q
echo       Done.
echo.

echo [2/3] Running experiment...
echo       You will see a progress bar for each project.
echo       Each project takes ~2-5 minutes and costs ~$0.50-1.50
echo.

python -m delphic_llm.evaluation.run_experiment ^
    --data data\nasa93.csv ^
    --n 3 ^
    --seeds 42 ^
    --conditions delphic_full b1 ^
    --output results\test\

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Experiment failed. See the error above.
    pause
    exit /b 1
)

echo.
echo [3/3] Generating paper table values...
python -m delphic_llm.evaluation.generate_tables --results results\test\

echo.
echo ============================================================
echo  Quick test complete.
echo  If results look reasonable, run run_experiments.bat
echo  for the full 50-project experiment.
echo ============================================================
echo.
pause
