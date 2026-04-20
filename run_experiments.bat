@echo off
REM DELPHIC-LLM: Full experiment runner for Windows
REM 50 projects, 3 seeds, 6 conditions -- ~$40-60 USD -- ~3-5 hours

echo.
echo ============================================================
echo  DELPHIC-LLM Full Experiment Runner
echo  50 projects x 3 seeds x 6 conditions
echo  Estimated: $40-60 USD,  3-5 hours unattended
echo ============================================================
echo.
echo TIP: You can minimise this window and come back later.
echo      Progress is saved after each seed so nothing is lost
echo      if the computer sleeps or restarts.
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
    echo Copy nasa93.csv into the data\ subfolder.
    echo.
    pause
    exit /b 1
)

if not exist "results" mkdir results

echo [1/3] Installing Python dependencies...
pip install openai pydantic sentence-transformers numpy pandas scipy scikit-learn -q
echo       Done.
echo.

echo [2/3] Running all conditions...
echo       Conditions: DELPHIC-LLM full, B1, B2, B3, ABL-1, ABL-2 + ABL-3 (20 projects)
echo       Each project costs ~$0.50-2.00 and takes ~2-5 minutes.
echo       Per-project progress will show below.
echo.

python -m delphic_llm.evaluation.run_experiment ^
    --data data\nasa93.csv ^
    --n 50 ^
    --seeds 42 43 44 ^
    --conditions delphic_full b1 b2 b3 abl1 abl2 ^
    --abl3_n 20 ^
    --output results\

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Experiment failed. Check the error above.
    echo Partial results may have been saved to results\
    pause
    exit /b 1
)

echo.
echo [3/3] Generating paper tables...
python -m delphic_llm.evaluation.generate_tables --results results\

echo.
echo ============================================================
echo  COMPLETE. Copy the values above into your Word document.
echo  Full results saved to: results\results_final.json
echo ============================================================
echo.
pause
