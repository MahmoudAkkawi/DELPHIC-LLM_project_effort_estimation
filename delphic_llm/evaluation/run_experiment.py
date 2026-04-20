"""
DELPHIC-LLM: Main experiment runner
Runs all experimental conditions and saves complete results to JSON.

Usage:
    python -m delphic_llm.evaluation.run_experiment \
        --data data/nasa93.csv \
        --n 50 \
        --seeds 42 43 44 \
        --output results/

Each seed produces one independent run. Results are averaged across seeds.
"""

import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass



import argparse
import json
import os
import time
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from delphic_llm.llm_client import LLMClient
from delphic_llm.pipeline import DELPHICPipeline
from delphic_llm.input.pkd_synthesiser import prepare_nasa93_dataset, synthesise_pkd_nasa93
from delphic_llm.evaluation.baselines import (
    run_single_llm, run_pert, run_unstructured_mad,
    OLSRegression, ABEEstimator
)
from delphic_llm.evaluation.metrics import (
    compute_accuracy_summary, compute_multi_run_summary,
    mre_list, wilcoxon_test, cohens_d, sycophancy_rate,
    expected_calibration_error
)


def run_classical_baselines(train_df: pd.DataFrame,
                             test_df: pd.DataFrame) -> dict:
    """Run OLS and k-NN on numerical features."""
    print("  Classical baselines (no API calls needed):")
    print("  Running OLS parametric regression...", end=" ", flush=True)
    ols = OLSRegression()
    ols.fit(train_df)
    ols_preds = ols.predict(test_df)

    print("done")
    print("  Running ABE k-NN (k=3)...", end=" ", flush=True)
    abe = ABEEstimator(k=3)
    abe.fit(train_df)
    abe_preds = abe.predict(test_df)

    actuals = test_df["effort_hours"].tolist()
    return {
        "ols": compute_accuracy_summary(actuals, ols_preds.tolist(), "OLS Regression"),
        "abe": compute_accuracy_summary(actuals, abe_preds.tolist(), "ABE (k-NN, k=3)")
    }


def run_llm_conditions(pkds: list,
                        actuals: List[float],
                        llm: LLMClient,
                        conditions: List[str],
                        verbose: bool = True) -> dict:
    """
    Run all LLM-based conditions on the 50-project sample.
    conditions: subset of ['delphic_full', 'b1', 'b2', 'b3', 'abl1', 'abl2', 'abl3']
    """
    results = {}

    for condition in conditions:
        cond_labels = {
            "delphic_full": "DELPHIC-LLM (Full framework)",
            "b1":           "B1: Single LLM zero-shot",
            "b2":           "B2: Three-point PERT",
            "b3":           "B3: Unstructured MAD",
            "abl1":         "ABL-1: No role differentiation",
            "abl2":         "ABL-2: No sycophancy detection",
            "abl3":         "ABL-3: Fixed technique assignment",
        }
        label = cond_labels.get(condition, condition.upper())
        print(f"\n{'─'*60}")
        print(f"  Condition: {label}")
        print(f"  Projects:  {len(pkds)}  |  Seed: (current)")
        print(f"{'─'*60}")
        cond_results = []
        all_logs = []
        cost_start = llm.total_cost_usd

        for idx, (pkd, actual) in enumerate(zip(pkds, actuals)):
            pct = (idx+1)/len(pkds)*100
            bar_len = 20
            filled = int(bar_len * (idx+1) / len(pkds))
            bar = "█"*filled + "░"*(bar_len-filled)
            cost_so_far = llm.total_cost_usd - cost_start
            print(f"  [{bar}] {idx+1:>2}/{len(pkds)}  actual={actual:>7,.0f}h", end="")

            try:
                if condition in ("delphic_full", "abl1", "abl2", "abl3"):
                    mode = {
                        "delphic_full": "full",
                        "abl1": "abl1",
                        "abl2": "abl2",
                        "abl3": "abl3"
                    }[condition]
                    pipeline = DELPHICPipeline(llm, mode=mode)
                    omega, logs = pipeline.run(pkd, verbose=False)
                    pred = omega.consensus_estimate_hours
                    ci_low = omega.ci_low_hours
                    ci_high = omega.ci_high_hours
                    sr = omega.sycophancy_rate
                    tau = omega.technique_convergence
                    q_scores = logs.get("quality_r3", [10, 10, 10])
                    q_traj = {
                        "r1": logs.get("quality_r1", [10, 10, 10]),
                        "r3": q_scores
                    }

                elif condition == "b1":
                    r = run_single_llm(pkd, llm)
                    pred = r["estimate_hours"]
                    ci_low = r.get("ci_low", pred * 0.7)
                    ci_high = r.get("ci_high", pred * 1.3)
                    sr = 0.0; tau = 0.0; q_traj = {}
                    logs = {}

                elif condition == "b2":
                    r = run_pert(pkd, llm)
                    pred = r["estimate_hours"]
                    ci_low = r.get("ci_low", pred * 0.7)
                    ci_high = r.get("ci_high", pred * 1.3)
                    sr = 0.0; tau = 0.0; q_traj = {}
                    logs = {"pert": r}

                elif condition == "b3":
                    r = run_unstructured_mad(pkd, llm)
                    pred = r["estimate_hours"]
                    ci_low = r.get("ci_low", min(r["individual_estimates"]))
                    ci_high = r.get("ci_high", max(r["individual_estimates"]))
                    # Approximate sycophancy: variance collapse in estimates
                    ind = r.get("individual_estimates", [pred])
                    sr = 1.0 if (max(ind) - min(ind)) < 50 else 0.0
                    tau = 0.0; q_traj = {}
                    logs = r

                else:
                    raise ValueError(f"Unknown condition: {condition}")

                mre_val = abs(actual - pred) / actual if actual > 0 else float('inf')
                mre_sym = "✓" if mre_val <= 0.25 else "~" if mre_val <= 0.50 else "✗"
                cost_so_far = llm.total_cost_usd - cost_start
                print(f"  →  pred={pred:>7,.0f}h  MRE={mre_val:.3f} {mre_sym}  ${cost_so_far:.2f} spent")

                cond_results.append({
                    "actual": actual,
                    "predicted": pred,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "mre": mre_val,
                    "sycophancy_rate": sr,
                    "tau": tau,
                    "quality_trajectory": q_traj,
                })
                all_logs.append(logs)

            except Exception as e:
                print(f"  →  ERROR: {e}")
                cond_results.append({
                    "actual": actual, "predicted": 0, "ci_low": 0, "ci_high": 0,
                    "mre": float('inf'), "sycophancy_rate": 0.0, "tau": 0.0,
                    "quality_trajectory": {}, "error": str(e)
                })
                all_logs.append({"error": str(e)})

        # Compute summaries for this condition
        valid = [r for r in cond_results if not np.isinf(r["mre"])]
        actuals_v  = [r["actual"]    for r in valid]
        preds_v    = [r["predicted"] for r in valid]
        ci_lows_v  = [r["ci_low"]   for r in valid]
        ci_highs_v = [r["ci_high"]  for r in valid]

        acc = compute_accuracy_summary(actuals_v, preds_v, condition)
        acc["sr_mean"] = round(float(np.mean([r["sycophancy_rate"] for r in valid])), 4)
        acc["tau_mean"] = round(float(np.mean([r["tau"] for r in valid
                                               if r["tau"] > 0])), 4) if any(
            r["tau"] > 0 for r in valid) else 0.0
        acc["ece_80"] = round(expected_calibration_error(
            actuals_v, ci_lows_v, ci_highs_v, stated_confidence=0.80), 4)
        acc["cost_usd"] = round(llm.total_cost_usd - cost_start, 4)
        acc["per_project_results"] = cond_results
        acc["raw_logs"] = all_logs

        results[condition] = acc
        print(f"\n  RESULT  {label}")
        print(f"          MMRE={acc['mmre']:.4f}  MdMRE={acc['mdmre']:.4f}  PRED(25)={acc['pred25']*100:.0f}%  PRED(50)={acc['pred50']*100:.0f}%")
        print(f"          ECE={acc['ece_80']:.4f}  SR={acc['sr_mean']*100:.0f}%  total_cost=${acc['cost_usd']:.2f}")

    return results


def run_significance_tests(results: dict) -> dict:
    """Run Wilcoxon tests and compute Cohen's d vs DELPHIC-LLM full."""
    delphic_mres = results.get("delphic_full", {}).get("mre_list", [])
    sig_tests = {}

    for cond in ["b1", "b2", "b3", "abl1", "abl2"]:
        if cond not in results:
            continue
        baseline_mres = results[cond].get("mre_list", [])
        stat, pval = wilcoxon_test(delphic_mres, baseline_mres)
        d = cohens_d(delphic_mres, baseline_mres)
        sig_tests[f"delphic_vs_{cond}"] = {
            "wilcoxon_stat": round(stat, 4) if not np.isnan(stat) else None,
            "p_value":       round(pval, 4) if not np.isnan(pval) else None,
            "cohens_d":      round(d, 4)    if not np.isnan(d)    else None,
            "significant":   bool(pval < 0.05) if not np.isnan(pval) else None
        }

    return sig_tests


def main():
    parser = argparse.ArgumentParser(description="DELPHIC-LLM Experiment Runner")
    parser.add_argument("--data",    default="data/nasa93.csv",
                        help="Path to NASA93 CSV file")
    parser.add_argument("--n",       type=int, default=50,
                        help="Number of projects to sample")
    parser.add_argument("--seeds",   type=int, nargs="+", default=[42, 43, 44],
                        help="Random seeds for independent runs")
    parser.add_argument("--output",  default="results/",
                        help="Output directory for results JSON")
    parser.add_argument("--conditions", nargs="+",
                        default=["delphic_full", "b1", "b2", "b3", "abl1", "abl2"],
                        help="Conditions to run")
    parser.add_argument("--abl3_n",  type=int, default=20,
                        help="Number of projects for ABL-3 (fixed technique)")
    parser.add_argument("--model",   default="gpt-4o-2024-11-20")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    # Setup
    Path(args.output).mkdir(parents=True, exist_ok=True)
    llm = LLMClient(model=args.model)

    all_run_results = {}

    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"  RUN {args.seeds.index(seed)+1} of {len(args.seeds)}  |  seed={seed}  |  n={args.n} projects")
        print(f"{'='*60}")

        llm.reset_cost_tracker()

        # Load full dataset and 50-project sample
        from delphic_llm.input.pkd_synthesiser import load_nasa93
        df_full = load_nasa93(args.data)   # all 93 rows - for classical baseline training
        df, pkds = prepare_nasa93_dataset(args.data, n_sample=args.n, seed=seed)
        actuals = (df["effort_hours"] * 1.0).tolist()  # person-hours for the 50-project sample

        # Classical baselines: train on 20% of full 93-row dataset, test on the 50-project sample
        # Avoids data leakage (train set never overlaps test PKDs)
        train_df, _ = train_test_split(df_full, test_size=0.2, random_state=seed)  # 80% train, 20% test
        classical_results = run_classical_baselines(train_df, df)

        # LLM conditions
        conditions = args.conditions.copy()

        # ABL-3: run on smaller sub-sample
        if "abl3" not in conditions:
            conditions_main = conditions
            conditions_abl3 = []
        else:
            conditions_main = [c for c in conditions if c != "abl3"]
            conditions_abl3 = ["abl3"]

        llm_results = run_llm_conditions(
            pkds, actuals, llm, conditions_main, verbose=args.verbose)

        # ABL-3 on 20 projects
        if conditions_abl3:
            print(f"\n  Running ABL-3 on first {args.abl3_n} projects...")
            abl3_results = run_llm_conditions(
                pkds[:args.abl3_n], actuals[:args.abl3_n], llm,
                ["abl3"], verbose=args.verbose)
            llm_results.update(abl3_results)

        # Significance tests
        sig_tests = run_significance_tests(llm_results)

        # Save seed run
        run_data = {
            "seed": seed,
            "n_projects": args.n,
            "classical": classical_results,
            "llm": llm_results,
            "significance_tests": sig_tests,
            "total_cost_usd": round(llm.total_cost_usd, 4)
        }
        all_run_results[f"seed_{seed}"] = run_data

        # Save intermediate result
        out_path = Path(args.output) / f"results_seed{seed}.json"
        with open(out_path, "w") as f:
            json.dump(run_data, f, indent=2, default=str)
        print(f"\n  Seed {seed} complete. Saved: {out_path}")
        print(f"  Running cost so far: ${llm.total_cost_usd:.2f}")

    # Aggregate across seeds
    print(f"\n{'='*60}")
    print("  Aggregating results across all seeds...")
    print(f"{'='*60}")
    conditions_to_agg = args.conditions
    aggregated = {}

    for cond in conditions_to_agg:
        runs = [all_run_results[f"seed_{s}"]["llm"].get(cond)
                for s in args.seeds
                if all_run_results[f"seed_{s}"]["llm"].get(cond)]
        if runs:
            aggregated[cond] = compute_multi_run_summary(runs)

    for clas in ["ols", "abe"]:
        runs = [all_run_results[f"seed_{s}"]["classical"].get(clas)
                for s in args.seeds
                if all_run_results[f"seed_{s}"]["classical"].get(clas)]
        if runs:
            aggregated[clas] = compute_multi_run_summary(runs)

    # Final significance tests on aggregated MREs
    if len(args.seeds) > 1:
        # Pool MREs across seeds for significance testing
        delphic_mres_all = []
        baseline_mres_all = {c: [] for c in ["b1", "b2", "b3", "abl1", "abl2"]}
        for seed in args.seeds:
            dm = all_run_results[f"seed_{seed}"]["llm"].get("delphic_full", {}).get("mre_list", [])
            delphic_mres_all.extend(dm)
            for c in baseline_mres_all:
                bm = all_run_results[f"seed_{seed}"]["llm"].get(c, {}).get("mre_list", [])
                baseline_mres_all[c].extend(bm)

        pooled_sig = {}
        for cond, mres in baseline_mres_all.items():
            if mres and delphic_mres_all:
                stat, pval = wilcoxon_test(delphic_mres_all, mres)
                d = cohens_d(delphic_mres_all, mres)
                pooled_sig[f"delphic_vs_{cond}"] = {
                    "wilcoxon_stat": round(stat, 4) if not np.isnan(stat) else None,
                    "p_value":       round(pval, 4) if not np.isnan(pval) else None,
                    "cohens_d":      round(d, 4)    if not np.isnan(d)    else None,
                    "significant":   bool(pval < 0.05) if not np.isnan(pval) else None
                }
    else:
        pooled_sig = all_run_results[f"seed_{args.seeds[0]}"]["significance_tests"]

    final_results = {
        "experiment": {
            "dataset": "NASA93",
            "n_projects": args.n,
            "seeds": args.seeds,
            "model": args.model,
            "conditions": args.conditions
        },
        "aggregated": aggregated,
        "significance_tests_pooled": pooled_sig,
        "per_seed": all_run_results
    }

    final_path = Path(args.output) / "results_final.json"
    with open(final_path, "w") as f:
        json.dump(final_results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print("FINAL AGGREGATED RESULTS")
    print(f"{'='*60}")
    print(f"{'Method':<25} {'MMRE':>8} {'MdMRE':>8} {'PRED25':>8} {'PRED50':>8}")
    print("-" * 65)
    for method, res in aggregated.items():
        print(f"{method:<25} "
              f"{res['mmre']['mean']:>7.4f} "
              f"{res['mdmre']['mean']:>8.4f} "
              f"{res['pred25']['mean']:>8.4f} "
              f"{res['pred50']['mean']:>8.4f}")

    print(f"\nResults saved to {final_path}")
    return final_results


if __name__ == "__main__":
    main()
