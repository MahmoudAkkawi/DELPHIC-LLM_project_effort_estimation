# DELPHIC-LLM: Full experiment runner (PowerShell version)
# More robust than .bat — use this if run_experiments.bat has issues
# Usage: Right-click → Run with PowerShell
#        OR: powershell -ExecutionPolicy Bypass -File run_experiments.ps1

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " DELPHIC-LLM Experiment Runner (PowerShell)" -ForegroundColor Cyan
Write-Host "============================================================"
Write-Host ""

# Check API key
if (-not $env:OPENAI_API_KEY) {
    Write-Host "ERROR: OPENAI_API_KEY not set." -ForegroundColor Red
    Write-Host ""
    Write-Host "Set it with:"
    Write-Host '  $env:OPENAI_API_KEY = "sk-..."'
    Write-Host ""
    Write-Host "Or set it permanently in System Properties > Environment Variables"
    Read-Host "Press Enter to exit"
    exit 1
}

# Check dataset
if (-not (Test-Path "data\nasa93.csv")) {
    Write-Host "ERROR: data\nasa93.csv not found." -ForegroundColor Red
    Write-Host "Place your NASA93 dataset at: $(Get-Location)\data\nasa93.csv"
    Read-Host "Press Enter to exit"
    exit 1
}

# Create results directory
New-Item -ItemType Directory -Force -Path "results" | Out-Null

# Install dependencies
Write-Host "Installing Python dependencies..." -ForegroundColor Yellow
pip install openai pydantic sentence-transformers numpy pandas scipy scikit-learn
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: pip install failed." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "Running full experiment (50 projects, 3 seeds)..." -ForegroundColor Green
Write-Host "Estimated: ~`$40-60 USD, 3-5 hours"
Write-Host ""

python -m delphic_llm.evaluation.run_experiment `
    --data "data\nasa93.csv" `
    --n 50 `
    --seeds 42 43 44 `
    --conditions delphic_full b1 b2 b3 abl1 abl2 `
    --abl3_n 20 `
    --output "results\"

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Experiment failed." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "Generating paper tables..." -ForegroundColor Green
python -m delphic_llm.evaluation.generate_tables --results "results\"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " DONE. Results: results\results_final.json" -ForegroundColor Green
Write-Host " Copy-paste values printed above for your Word document." -ForegroundColor Green
Write-Host "============================================================"
Read-Host "Press Enter to close"
