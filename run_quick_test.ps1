# DELPHIC-LLM: Quick test (PowerShell)
# Usage: powershell -ExecutionPolicy Bypass -File run_quick_test.ps1

Write-Host "DELPHIC-LLM Quick Test (3 projects, ~`$3-5)" -ForegroundColor Cyan

if (-not $env:OPENAI_API_KEY) {
    Write-Host 'ERROR: Set $env:OPENAI_API_KEY = "sk-..."' -ForegroundColor Red
    Read-Host "Press Enter"; exit 1
}
if (-not (Test-Path "data\nasa93.csv")) {
    Write-Host "ERROR: Place nasa93.csv at data\nasa93.csv" -ForegroundColor Red
    Read-Host "Press Enter"; exit 1
}

New-Item -ItemType Directory -Force -Path "results\test" | Out-Null
pip install openai pydantic sentence-transformers numpy pandas scipy scikit-learn -q

python -m delphic_llm.evaluation.run_experiment `
    --data "data\nasa93.csv" `
    --n 3 --seeds 42 `
    --conditions delphic_full b1 `
    --output "results\test\"

python -m delphic_llm.evaluation.generate_tables --results "results\test\"
Write-Host "Done. Run run_experiments.ps1 for full experiment." -ForegroundColor Green
Read-Host "Press Enter"
