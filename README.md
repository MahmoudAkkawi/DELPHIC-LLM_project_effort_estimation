# DELPHIC-LLM: Hierarchically Orchestrated Multi-Agent Framework for Confidence-Aware Project Effort Estimation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-IJISPM-green.svg)](link-to-paper)

A hierarchically orchestrated multi-agent framework that renders structured Delphi deliberation computationally accessible for project effort estimation across all domains, whilst adding capabilities unavailable to human expert panels.

---

## 📋 Overview

DELPHIC-LLM addresses the longstanding challenge of accurate project effort estimation by combining:

- **Hierarchical governance**: An Orchestrator Agent (OA) evaluates argument quality, detects sycophancy, monitors convergence, and synthesises consensus
- **Disposition-engineered PM-Expert agents**: Three agents with distinct reasoning philosophies (Risk-Aware Conservative, Delivery-Focused Balanced, Efficiency-Oriented Optimising) grounded in PMBOK, PRINCE2, ISO 21502, and Agile
- **Role-differentiated adversarial review**: Challenger, Builder, and Risk Analyst roles ensure genuine intellectual conflict
- **Calibrated confidence profiles**: Four-axis confidence (C_est, C_assump, C_comp, C_cons) derived from deliberation dynamics, not model self-report
- **Complete decision-support output (Ω)**: Consensus estimate + 80% CI, confidence profile, argument graph, risk register, and recommendations

---

## 🎯 Key Results

Evaluated on the NASA93 benchmark (n=50 per seed, 3 independent seeds):

| Metric | DELPHIC-LLM | B2: PERT | B3: MAD | Improvement |
|--------|-------------|----------|---------|-------------|
| **MMRE** | 0.819 ± 0.248 | 0.989 ± 0.125 | 0.958 ± 0.207 | **−17.2%** vs PERT<br>**−14.6%** vs MAD |
| **MdMRE** | 0.510 | 0.557 | 0.509 | Competitive |
| **PRED(25)** | 31.6% | 27.3% | 31.3% | +4.3pp vs PERT |
| **PRED(50)** | 49.6% | 43.3% | 46.7% | +6.3pp vs PERT |

**Deliberation Quality:**
- Sycophancy rate: 13.9% ± 2.4% (consistent across seeds)
- Ablation: removing sycophancy detection increases MMRE by 4.2%
- Technique convergence τ = 0.422 (58% technique diversity)
- Well-calibrated confidence: C_est=0.357, C_assump=0.895, C_comp=0.248, C_cons=0.639

**Economics:**
- **$0.119 per decision pack** (≈£0.09)
- **>99% cost reduction** vs facilitated human Delphi sessions (£9,720–£54,000)

---

## 🚀 Quick Start

### Prerequisites

```bash
python >= 3.10
pip install openai pandas numpy scikit-learn sentence-transformers tqdm
```

### Installation

```bash
git clone https://github.com/MahmoudAkkawi/DELPHIC-LLM_project_effort_estimation.git
cd delphic-llm
pip install -r requirements.txt
```

### Basic Usage

```python
from delphic_llm import DELPHIC_Framework

# Initialize framework
framework = DELPHIC_Framework(
    api_key="your-openai-api-key",
    model="gpt-4o-2024-11-20"
)

# Prepare project knowledge document
pkd = {
    'context': 'Web-based customer portal for e-commerce',
    'scope': 'User authentication, product catalog, shopping cart, payment integration',
    'team': '3 senior developers, 1 junior, 1 QA',
    'artefacts': None,  # Optional: diagrams, mockups
    'analogues': None,  # Optional: similar past projects
    'risks': 'Third-party payment API stability uncertain'
}

# Run estimation
result = framework.estimate(pkd)

# Access outputs
print(f"Consensus estimate: {result['consensus_estimate']:.0f} person-hours")
print(f"80% CI: [{result['ci_lower']:.0f}, {result['ci_upper']:.0f}]")
print(f"Confidence profile: {result['confidence']}")
print(f"Sycophancy detected: {result['sycophancy_rate']:.1%}")
print(f"Technique convergence τ: {result['tau']:.3f}")

# Full decision pack
print(result['risk_register'])
print(result['recommendations'])
```
---

## 🔬 Experimental Design

### Dataset
- **NASA93**: 93 historical NASA software projects (1971–1987)
- Effort range: 1,277 – 1,248,072 person-hours (3 orders of magnitude)
- Stratified sampling: n=50 per seed, proportional across 14 application types
- 3 independent seeds (42, 43, 44)

### Baselines
- **OLS regression**: Parametric model on COCOMO features
- **ABE (k-NN, k=3)**: Analogy-based estimation
- **B1**: Single LLM (GPT-4o), zero-shot
- **B2**: Three-point PERT via LLM (PMBOK formula)
- **B3**: Unstructured multi-agent debate (Du et al., 2023)
- **ABL-1**: DELPHIC-LLM without role differentiation
- **ABL-2**: DELPHIC-LLM without sycophancy detection

### Metrics
- **MMRE**: Mean Magnitude of Relative Error (primary)
- **MdMRE**: Median MRE (robustness to outliers)
- **PRED(25)** / **PRED(50)**: Proportion within 25% / 50% of actual
- **Wilcoxon signed-rank test** (two-tailed, α=0.05)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│  INPUT: Project Knowledge Document (PKD)                │
│  PKD₁ context · PKD₂ scope · PKD₃ team · PKD₄ artefacts│
│  PKD₅ analogues · PKD₆ risks                            │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────────────────────┐
│  TIER 1: Orchestrator Agent (OA)                        │
│  ┌──────────┬──────────┬──────────┬──────────┐         │
│  │ Argument │Sycophancy│Convergence│ Quality- │         │
│  │ Quality  │Detection │Monitoring │ Weighted │         │
│  │ Scoring  │ (embed.  │   (κ)     │Synthesis │         │
│  │ (Q₁–Q₅)  │+ shift)  │           │ (G, Ω)   │         │
│  └──────────┴──────────┴──────────┴──────────┘         │
└─────────┬──────────────┬──────────────┬────────────────┘
          │              │              │
          ↓              ↓              ↓
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  D₁: AI     │  │  D₂: AI     │  │  D₃: AI     │
│  Expert     │  │  Expert     │  │  Expert     │
│  Agent      │  │  Agent      │  │  Agent      │
├─────────────┤  ├─────────────┤  ├─────────────┤
│ Risk-Aware  │  │ Delivery-   │  │ Efficiency- │
│ Conservative│  │ Focused     │  │ Oriented    │
│             │  │ Balanced    │  │ Optimising  │
├─────────────┤  ├─────────────┤  ├─────────────┤
│ R2 Role:    │  │ R2 Role:    │  │ R2 Role:    │
│ Challenger  │  │ Builder     │  │ Risk Analyst│
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                │                │
       └────────────────┴────────────────┘
                        │
                ┌───────┴───────┐
                │  Shared KB K  │
                │  PMBOK·PRINCE2│
                │  ISO·Agile    │
                └───────────────┘
                        │
                        ↓
          ┌─────────────────────────────┐
          │  Output: Decision Pack Ω    │
          │  • Consensus ê + 80% CI     │
          │  • Confidence (4 axes)      │
          │  • Argument graph G=(V,E)   │
          │  • Risk register            │
          │  • Recommendations          │
          │  • Technique signal τ       │
          └─────────────────────────────┘
```

---

## 📊 Reproducing Results

### Run Full Experiment

```bash
cd experiments
python run_experiment.py --seeds 42 43 44 --n_per_seed 50
```

This will:
1. Load NASA93 dataset
2. Run stratified sampling for each seed
3. Execute all 8 conditions (OLS, ABE, B1, B2, B3, ABL-1, ABL-2, DELPHIC-LLM)
4. Compute metrics (MMRE, MdMRE, PRED)
5. Save results to `results/results_seed{N}.json`
6. Generate aggregated statistics in `results/results_final.json`

**Expected runtime**: ~45 minutes per seed on standard hardware (depends on API rate limits)

**Expected cost**: ~$22 per seed (total ~$65 for 3 seeds)

### Analyze Results

```python
import json
import pandas as pd

# Load results
with open('results/results_final.json') as f:
    results = json.load(f)

# Extract MMRE comparison
methods = ['OLS', 'ABE', 'B1', 'B2', 'B3', 'ABL-1', 'ABL-2', 'DELPHIC-LLM']
mmre = [results['classical']['ols']['mmre'],
        results['classical']['abe']['mmre'],
        results['llm']['b1']['mmre'],
        results['llm']['b2']['mmre'],
        results['llm']['b3']['mmre'],
        results['llm']['abl1']['mmre'],
        results['llm']['abl2']['mmre'],
        results['llm']['delphic_full']['mmre']]

df = pd.DataFrame({'Method': methods, 'MMRE': mmre})
print(df)
```

---

## 🔧 Configuration

Key parameters in `config.yaml`:

```yaml
model:
  name: "gpt-4o-2024-11-20"
  temperature: 0.7  # For agents
  orchestrator_temperature: 0.2

deliberation:
  max_rounds: 5
  convergence_threshold: 0.15  # κ threshold
  sycophancy_similarity_threshold: 0.90
  sycophancy_shift_threshold: 0.10
  revision_cap: 0.15  # ±15% in R3

embedding:
  model: "all-mpnet-base-v2"

confidence:
  axes: ["C_est", "C_assump", "C_comp", "C_cons"]
```
## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

---

## 🔗 Links

- **Dataset**: NASA93 available at https://github.com/timm/ourmine/blob/master/our/arffs/effest/nasa93.arff
- **Issues**: [GitHub Issues](https://github.com/yourusername/delphic-llm/issues)

---

**Note**: This is a research implementation. For production use in critical projects, please conduct thorough validation on your domain-specific data.
