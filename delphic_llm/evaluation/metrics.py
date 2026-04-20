"""
DELPHIC-LLM: Evaluation metrics
Implements all metrics from Section 4.3 of the paper exactly.
"""
import numpy as np
from statistics import median, mean, stdev
from typing import List, Optional, Tuple
from scipy import stats


# ── Accuracy metrics ──────────────────────────────────────────────────────────

def mre(actual: float, predicted: float) -> float:
    """MRE_i = |y_i - ŷ_i| / y_i"""
    if actual == 0:
        return float('inf')
    return abs(actual - predicted) / actual


def mmre(actuals: List[float], predictions: List[float]) -> float:
    """MMRE = (1/N) Σ MRE_i"""
    mres = [mre(a, p) for a, p in zip(actuals, predictions)]
    valid = [m for m in mres if not np.isinf(m)]
    return mean(valid) if valid else float('inf')


def mdmre(actuals: List[float], predictions: List[float]) -> float:
    """MdMRE = median of MRE values (robust to outliers)."""
    mres = [mre(a, p) for a, p in zip(actuals, predictions)]
    valid = [m for m in mres if not np.isinf(m)]
    return median(valid) if valid else float('inf')


def pred(actuals: List[float], predictions: List[float],
         threshold: float = 0.25) -> float:
    """PRED(threshold) = proportion of estimates within threshold of actuals."""
    hits = sum(1 for a, p in zip(actuals, predictions)
               if mre(a, p) <= threshold)
    return hits / len(actuals) if actuals else 0.0


def pred25(actuals, predictions) -> float:
    return pred(actuals, predictions, 0.25)


def pred50(actuals, predictions) -> float:
    return pred(actuals, predictions, 0.50)


def mre_list(actuals: List[float], predictions: List[float]) -> List[float]:
    """Return list of individual MRE values."""
    return [mre(a, p) for a, p in zip(actuals, predictions)]


# ── Statistical significance ──────────────────────────────────────────────────

def wilcoxon_test(mres_a: List[float],
                   mres_b: List[float]) -> Tuple[float, float]:
    """
    Wilcoxon signed-rank test (paired, two-tailed).
    Returns (statistic, p_value).
    a = DELPHIC-LLM, b = baseline.
    H0: median difference = 0.
    """
    # Filter paired valid values
    pairs = [(a, b) for a, b in zip(mres_a, mres_b)
             if not (np.isinf(a) or np.isinf(b))]
    if len(pairs) < 5:
        return float('nan'), float('nan')
    arr_a = np.array([p[0] for p in pairs])
    arr_b = np.array([p[1] for p in pairs])
    try:
        stat, pval = stats.wilcoxon(arr_a, arr_b, alternative='two-sided')
        return float(stat), float(pval)
    except Exception:
        return float('nan'), float('nan')


def cohens_d(mres_a: List[float], mres_b: List[float]) -> float:
    """Cohen's d effect size for two MRE distributions."""
    valid_a = [m for m in mres_a if not np.isinf(m)]
    valid_b = [m for m in mres_b if not np.isinf(m)]
    if len(valid_a) < 2 or len(valid_b) < 2:
        return float('nan')
    pooled_std = np.sqrt((np.std(valid_a, ddof=1)**2 +
                          np.std(valid_b, ddof=1)**2) / 2)
    if pooled_std == 0:
        return 0.0
    return (mean(valid_a) - mean(valid_b)) / pooled_std


# ── Confidence calibration ────────────────────────────────────────────────────

def expected_calibration_error(actuals: List[float],
                                ci_lows: List[float],
                                ci_highs: List[float],
                                stated_confidence: float = 0.80,
                                n_bins: int = 10) -> float:
    """
    ECE = Σ_{m=1}^{M} (|B_m|/N) × |acc(B_m) - conf(B_m)|
    For interval-based predictions with a single stated confidence level.
    """
    if not actuals:
        return float('nan')

    # Hit = 1 if actual falls within CI
    hits = [int(lo <= a <= hi) for a, lo, hi in zip(actuals, ci_lows, ci_highs)]
    empirical_coverage = mean(hits)

    # With a single stated confidence level, ECE simplifies to:
    # |empirical_coverage - stated_confidence|
    return abs(empirical_coverage - stated_confidence)


def ece_by_bins(actuals: List[float],
                ci_lows: List[float],
                ci_highs: List[float],
                confidence_scores: List[float],
                n_bins: int = 10) -> float:
    """
    Full ECE with variable confidence scores per prediction.
    Used for DELPHIC-LLM's four-axis profile scores.
    """
    if not actuals or len(actuals) != len(confidence_scores):
        return float('nan')

    hits = [int(lo <= a <= hi)
            for a, lo, hi in zip(actuals, ci_lows, ci_highs)]

    # Sort by confidence score into bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for m in range(n_bins):
        bin_lo, bin_hi = bin_edges[m], bin_edges[m + 1]
        bin_mask = [bin_lo <= c < bin_hi for c in confidence_scores]
        bin_hits  = [h for h, m_ in zip(hits, bin_mask) if m_]
        bin_confs = [c for c, m_ in zip(confidence_scores, bin_mask) if m_]
        if not bin_hits:
            continue
        acc_bin  = mean(bin_hits)
        conf_bin = mean(bin_confs)
        ece += (len(bin_hits) / len(actuals)) * abs(acc_bin - conf_bin)

    return ece


# ── Deliberation quality metrics ──────────────────────────────────────────────

def sycophancy_rate(flags: list) -> float:
    """SR = proportion of revisions classified as sycophantic."""
    if not flags:
        return 0.0
    interventions = sum(1 for f in flags if getattr(f, 'intervention_sent', False))
    return interventions / max(len(flags), 1)


# ── Summary results dict ──────────────────────────────────────────────────────

def compute_accuracy_summary(actuals: List[float],
                               predictions: List[float],
                               method_name: str = "") -> dict:
    """Return complete accuracy summary dict."""
    mres = mre_list(actuals, predictions)
    return {
        "method":   method_name,
        "n":        len(actuals),
        "mmre":     round(mmre(actuals, predictions), 4),
        "mdmre":    round(mdmre(actuals, predictions), 4),
        "pred25":   round(pred25(actuals, predictions), 4),
        "pred50":   round(pred50(actuals, predictions), 4),
        "mre_list": [round(m, 4) for m in mres],
    }


def compute_multi_run_summary(run_results: list) -> dict:
    """
    Given results from multiple runs (seeds), compute mean ± SD.
    run_results: list of dicts each with 'mmre', 'mdmre', 'pred25', 'pred50'.
    """
    def _ms(key):
        vals = [r[key] for r in run_results if not np.isinf(r[key])]
        if not vals:
            return {"mean": float('nan'), "sd": float('nan')}
        return {"mean": round(mean(vals), 4),
                "sd":   round(stdev(vals) if len(vals) > 1 else 0.0, 4)}

    return {
        "n_runs":  len(run_results),
        "mmre":    _ms("mmre"),
        "mdmre":   _ms("mdmre"),
        "pred25":  _ms("pred25"),
        "pred50":  _ms("pred50"),
    }
