"""
DELPHIC-LLM: Baseline implementations
B1: Single LLM (GPT-4o, zero-shot)
B2: Three-point PERT via LLM
B3: Unstructured multi-agent debate (Du et al. framework)
Classical: OLS parametric regression, Analogy-based (k-NN)
"""
import json
import numpy as np
from typing import List, Optional
from pydantic import BaseModel

from delphic_llm.llm_client import LLMClient
from delphic_llm.models import ProjectKnowledgeDocument
from delphic_llm.prompts.templates import PM_KNOWLEDGE_BASE


# ─── B1: Single LLM ──────────────────────────────────────────────────────────

class SingleLLMOutput(BaseModel):
    estimate_hours: float
    technique_used: str
    reasoning: str
    confidence_interval_low: float
    confidence_interval_high: float


SINGLE_LLM_PROMPT = """You are a fully qualified project management expert.

PROJECT:
{pkd}

Estimate the total effort in person-hours to complete this project.
Respond ONLY with valid JSON:
{{
  "estimate_hours": <number>,
  "technique_used": "<technique name>",
  "reasoning": "<brief step-by-step reasoning>",
  "confidence_interval_low": <80% CI lower bound>,
  "confidence_interval_high": <80% CI upper bound>
}}"""


def run_single_llm(pkd: ProjectKnowledgeDocument, llm: LLMClient) -> dict:
    """B1: Single LLM zero-shot estimation."""
    user = SINGLE_LLM_PROMPT.format(pkd=pkd.to_context_string())
    result = llm.orchestrator_call(
        system=PM_KNOWLEDGE_BASE,
        user=user
    )
    if not result:
        simple_prompt = "Estimate total effort in person-hours. Respond ONLY with JSON: {\"estimate_hours\": <number>}"
        result = llm.orchestrator_call(system=PM_KNOWLEDGE_BASE, user=simple_prompt)
    if result is None:
        result = {}
    if isinstance(result, dict):
        # Normalise common LLM field name variations
        aliases = {
            "estimate": "estimate_hours", "totalEffort": "estimate_hours",
            "effortHours": "estimate_hours", "estimatedHours": "estimate_hours",
            "effort": "estimate_hours", "total_effort": "estimate_hours",
            "confidenceIntervalLow": "confidence_interval_low",
            "lowerBound": "confidence_interval_low", "lower_bound": "confidence_interval_low",
            "confidenceIntervalHigh": "confidence_interval_high",
            "upperBound": "confidence_interval_high", "upper_bound": "confidence_interval_high",
            "techniqueUsed": "technique_used", "technique": "technique_used",
        }
        r = {aliases.get(k, k): v for k, v in result.items()}
        est = r.get("estimate_hours", 0)
        if not est:
            for k in ["estimate", "effort_hours", "total_hours", "hours"]:
                if k in r and r[k]:
                    est = r[k]; break
        try:
            est = float(str(est).replace(",", "").split()[0]) if est else 0.0
        except (ValueError, AttributeError):
            est = 0.0
        ci_low  = float(r.get("confidence_interval_low",  est * 0.7) or est * 0.7)
        ci_high = float(r.get("confidence_interval_high", est * 1.4) or est * 1.4)
        return {"estimate_hours": est, "ci_low": ci_low, "ci_high": ci_high,
                "technique": r.get("technique_used", "unspecified")}


# ─── B2: Three-point PERT ─────────────────────────────────────────────────────

PERT_PROMPT = """You are a project estimation expert using Three-Point PERT estimation.

PROJECT:
{pkd}

Provide three estimates in person-hours:
- Optimistic (O): best case if everything goes well
- Most likely (M): most realistic estimate
- Pessimistic (P): worst case with major problems

Then compute: E = (O + 4*M + P) / 6, SD = (P - O) / 6

Respond ONLY with valid JSON:
{{
  "optimistic_hours": <O>,
  "most_likely_hours": <M>,
  "pessimistic_hours": <P>,
  "pert_estimate_hours": <E>,
  "std_dev_hours": <SD>,
  "reasoning": "<brief justification>"
}}"""


def run_pert(pkd: ProjectKnowledgeDocument, llm: LLMClient) -> dict:
    """B2: Three-point PERT estimation."""
    user = PERT_PROMPT.format(pkd=pkd.to_context_string())
    result = llm.orchestrator_call(
        system=PM_KNOWLEDGE_BASE,
        user=user
    )
    if isinstance(result, dict):
        O = float(result.get("optimistic_hours", 100))
        M = float(result.get("most_likely_hours", 200))
        P = float(result.get("pessimistic_hours", 400))
        E = (O + 4*M + P) / 6
        SD = (P - O) / 6
        return {
            "estimate_hours": round(E, 1),
            "ci_low": max(0, round(E - 1.28 * SD, 1)),  # 80% CI ≈ ±1.28 SD
            "ci_high": round(E + 1.28 * SD, 1),
            "O": O, "M": M, "P": P, "SD": round(SD, 1)
        }
    return {"estimate_hours": 0, "ci_low": 0, "ci_high": 0}


# ─── B3: Unstructured Multi-Agent Debate ─────────────────────────────────────

MAD_ROUND_PROMPT = """You are a project management expert estimating total effort.

PROJECT:
{pkd}

PREVIOUS ESTIMATES FROM PEERS:
{peer_estimates}

Based on the project description and your peers' perspectives, provide your updated 
effort estimate. You may agree, disagree, or synthesise as you see fit.

Respond ONLY with valid JSON:
{{
  "estimate_hours": <number>,
  "reasoning": "<your reasoning>"
}}"""


def run_unstructured_mad(pkd: ProjectKnowledgeDocument,
                          llm: LLMClient,
                          n_agents: int = 3,
                          n_rounds: int = 3) -> dict:
    """
    B3: Unstructured multi-agent debate (Du et al. framework).
    Agents freely exchange estimates for 3 rounds without roles or orchestrator.
    """
    # Round 1: independent estimates
    estimates = []
    for _ in range(n_agents):
        result = llm.orchestrator_call(
            system=PM_KNOWLEDGE_BASE,
            user=MAD_ROUND_PROMPT.format(
                pkd=pkd.to_context_string(),
                peer_estimates="(No peer estimates yet — this is Round 1)"
            )
        )
        if isinstance(result, dict):
            estimates.append(float(result.get("estimate_hours", 500)))
        else:
            estimates.append(500.0)

    # Rounds 2 and 3: free exchange
    for _ in range(n_rounds - 1):
        new_estimates = []
        peer_str = "\n".join([f"- Agent {i+1}: {e:.0f} hours"
                               for i, e in enumerate(estimates)])
        for _ in range(n_agents):
            result = llm.orchestrator_call(
                system=PM_KNOWLEDGE_BASE,
                user=MAD_ROUND_PROMPT.format(
                    pkd=pkd.to_context_string(),
                    peer_estimates=peer_str
                )
            )
            if isinstance(result, dict):
                new_estimates.append(float(result.get("estimate_hours", 500)))
            else:
                new_estimates.append(estimates[-1])
        estimates = new_estimates

    # Final: unweighted mean
    final_estimate = float(np.mean(estimates))
    return {
        "estimate_hours": round(final_estimate, 1),
        "individual_estimates": [round(e, 1) for e in estimates],
        "ci_low": round(min(estimates), 1),
        "ci_high": round(max(estimates), 1)
    }


# ─── Classical: OLS Parametric Regression ────────────────────────────────────

class OLSRegression:
    """
    Parametric regression baseline.
    Trained on 80/20 split of NASA93 numerical features.
    """
    def __init__(self):
        self.coef_  = None
        self.intercept_ = None
        self.feature_cols = [
            "rely", "data", "cplx", "time", "stor", "virt", "turn",
            "acap", "aexp", "pcap", "vexp", "lexp", "modp", "tool", "sced", "equivphyskloc"
        ]


    def _to_matrix(self, df):
        import pandas as pd
        label_map = {'vl':1,'l':2,'n':3,'h':4,'vh':5,'xh':6}
        cat_cols = [c for c in self.feature_cols if df[c].dtype == object]
        num_cols = [c for c in self.feature_cols if df[c].dtype != object]
        parts = []
        if cat_cols:
            parts.append(df[cat_cols].apply(lambda col: col.map(label_map)).fillna(3).astype(float))
        if num_cols:
            parts.append(df[num_cols].astype(float).fillna(df[num_cols].median()))
        import pandas as pd
        return pd.concat(parts, axis=1).values if parts else df[self.feature_cols].fillna(3.0).values

    def fit(self, df_train):
        from sklearn.linear_model import LinearRegression
        import numpy as np
        X = self._to_matrix(df_train)
        y = np.log(df_train["effort_hours"].values + 1)
        model = LinearRegression()
        model.fit(X, y)
        self.coef_      = model.coef_
        self.intercept_ = model.intercept_
        self._sklearn_model = model
        return self

    def predict(self, df) -> np.ndarray:
        import numpy as np
        X = self._to_matrix(df)
        log_pred = self._sklearn_model.predict(X)
        return np.exp(log_pred) - 1


# ─── Classical: Analogy-Based Estimation (k-NN) ───────────────────────────────

class ABEEstimator:
    """
    Analogy-Based Estimation using k-NN.
    k=3, Euclidean distance on COCOMO attributes.
    """
    def __init__(self, k: int = 3):
        self.k = k
        self._X_train = None
        self._y_train = None
        self.feature_cols = [
            "rely", "data", "cplx", "time", "stor", "virt", "turn",
            "acap", "aexp", "pcap", "vexp", "lexp", "modp", "tool", "sced"
        ]


    def _to_matrix(self, df):
        label_map = {'vl':1,'l':2,'n':3,'h':4,'vh':5,'xh':6}
        cat_cols = [c for c in self.feature_cols if df[c].dtype == object]
        num_cols = [c for c in self.feature_cols if df[c].dtype != object]
        parts = []
        if cat_cols:
            parts.append(df[cat_cols].apply(lambda col: col.map(label_map)).fillna(3).astype(float))
        if num_cols:
            parts.append(df[num_cols].astype(float).fillna(df[num_cols].median()))
        import pandas as pd
        return pd.concat(parts, axis=1).values if parts else df[self.feature_cols].fillna(3.0).values

    def fit(self, df_train):
        import numpy as np
        self._X_train = self._to_matrix(df_train)
        self._y_train = df_train["effort_hours"].values
        return self

    def predict(self, df) -> np.ndarray:
        import numpy as np
        X_test = self._to_matrix(df)
        predictions = []
        for x in X_test:
            dists = np.linalg.norm(self._X_train - x, axis=1)
            k_idx = np.argsort(dists)[:self.k]
            predictions.append(float(np.mean(self._y_train[k_idx])))
        return np.array(predictions)
