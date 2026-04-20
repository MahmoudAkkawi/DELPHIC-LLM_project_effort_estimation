# DELPHIC-LLM: Implementation

## Setup

### Windows

1. **Install Python 3.10+** from https://python.org — tick "Add Python to PATH" during install

2. **Open Command Prompt** (Win+R → cmd) or PowerShell in the project folder

3. **Install dependencies:**
   ```
   pip install openai pydantic sentence-transformers numpy pandas scipy scikit-learn
   ```

4. **Set your API key** (Command Prompt):
   ```
   set OPENAI_API_KEY=sk-...
   ```
   Or PowerShell:
   ```powershell
   $env:OPENAI_API_KEY = "sk-..."
   ```

5. **Place dataset:**
   ```
   mkdir data
   copy nasa93.csv data\nasa93.csv
   ```

6. **Run:**
   ```
   run_quick_test.bat        (verify setup, ~$3-5)
   run_experiments.bat       (full experiment, ~$40-60)
   ```
   If `.bat` has issues, use PowerShell instead:
   ```
   powershell -ExecutionPolicy Bypass -File run_experiments.ps1
   ```

### Linux / Mac

```bash
export OPENAI_API_KEY="sk-..."
pip install -r requirements.txt
cp nasa93.csv data/nasa93.csv
bash run_quick_test.sh
bash run_experiments.sh
```

---

## NASA93 Dataset — Verified Column Reference

The code is verified against the actual nasa93.csv (93 rows, 26 columns):

| Column | Role | Notes |
|--------|------|-------|
| `act_effort` | Target: effort (person-months × 152 = hours) | Main target |
| `equivphyskloc` | Size (KLOC) | NOT 'loc' |
| `rely/data/cplx/...` | 15 COCOMO cost drivers | Labels: vl/l/n/h/vh/xh |
| `cat2` | Project type | 14 categories e.g. 'avionicsmonitoring' |
| `center` | NASA centre (1/2/3/5/6) | Maps to facility name |
| `year` | Project year | 1971–1987 |
| `mode` | COCOMO mode | organic/semidetached/embedded |
| `app_type` | NOT USED | Always '0' — ignore |

---

## Running a Single Project Manually

```python
from delphic_llm.llm_client import LLMClient
from delphic_llm.pipeline import DELPHICPipeline
from delphic_llm.models import ProjectKnowledgeDocument

llm = LLMClient()
pipeline = DELPHICPipeline(llm, mode="full")

pkd = ProjectKnowledgeDocument(
    pkd1_context="NASA avionics monitoring software. Johnson Space Center, 1979.",
    pkd2_scope="High reliability, high complexity. ~25 KLOC. Semidetached mode.",
    pkd3_resources="Nominal analyst capability, high language experience, nominal tools.",
    pkd6_known_unknowns="Requirements completeness not fully specified."
)

omega, logs = pipeline.run(pkd, verbose=True)
print(f"Estimate: {omega.consensus_estimate_hours:.0f}h "
      f"[{omega.ci_low_hours:.0f}–{omega.ci_high_hours:.0f}h]")
print(f"Cost: ${logs['api_cost_usd']:.3f} | Time: {logs['time_seconds']:.1f}s")
for rec in omega.recommendations:
    print(f"  → {rec}")
```

---

## Confirmed Baseline Numbers (actual data, seed=42, n=50)

| Baseline | MMRE | PRED(25) | PRED(50) |
|----------|------|----------|----------|
| OLS regression | 0.714 | 16% | 50% |
| ABE (k-NN, k=3) | 0.814 | 38% | 78% |

---

## File Structure

```
delphic_llm/
├── models.py              # Pydantic schemas
├── llm_client.py          # OpenAI API wrapper
├── orchestrator.py        # OA: quality scoring, argument graph, sycophancy
├── pipeline.py            # Algorithm 1: deliberation protocol
├── agents/
│   └── pm_expert.py       # PM-Expert agent (D1/D2/D3)
├── prompts/
│   └── templates.py       # All prompts
├── input/
│   └── pkd_synthesiser.py # NASA93 → PKD
└── evaluation/
    ├── baselines.py        # B1 (Single LLM), B2 (PERT), B3 (MAD), OLS, k-NN
    ├── metrics.py          # MMRE, MdMRE, PRED(25), ECE, Wilcoxon, Cohen's d
    ├── run_experiment.py   # Main runner
    └── generate_tables.py  # Results → paper copy-paste values
```

---

## Cost Estimate

| Run | Projects | Cost |
|-----|----------|------|
| Quick test | 3 × 1 seed | ~$3–5 |
| Full experiment | 50 × 3 seeds, 6 conditions | ~$40–60 |
