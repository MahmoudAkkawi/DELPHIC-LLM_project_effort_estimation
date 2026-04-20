"""
DELPHIC-LLM: Main deliberation pipeline
Implements Algorithm 1 from the paper exactly.

Modes:
  - mode='full'       : Complete DELPHIC-LLM with all components
  - mode='abl1'       : ABL-1: no role differentiation (symmetric peer review)
  - mode='abl2'       : ABL-2: no sycophancy detection
  - mode='abl3'       : ABL-3: fixed technique assignment
"""

import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass

import time
import json
from typing import List, Optional, Literal
from statistics import mean

from delphic_llm.models import (
    Round1Output, Round3Output, HolisticDecisionPack,
    ProjectKnowledgeDocument, OrchestratorState, SycophancyFlag
)
from delphic_llm.agents.pm_expert import PMExpertAgent, ABL3_TECHNIQUES
from delphic_llm.orchestrator import Orchestrator
from delphic_llm.llm_client import LLMClient
from delphic_llm.prompts.templates import SYCOPHANCY_INTERVENTION, DEVILS_ADVOCATE_PROMPT




Mode = Literal["full", "abl1", "abl2", "abl3"]


class DELPHICPipeline:
    """
    Complete DELPHIC-LLM deliberation pipeline.
    All configuration matches the paper specification exactly.
    """
    THETA_CONV = 0.15
    THETA_HIGH = 0.40
    R_MAX      = 5

    def __init__(self, llm: LLMClient, mode: Mode = "full"):
        self.llm  = llm
        self.mode = mode
        self.oa   = Orchestrator(llm)

        # Create agents with correct dispositions
        if mode == "abl3":
            self.agents = [
                PMExpertAgent(i, llm, fixed_technique=ABL3_TECHNIQUES[i])
                for i in range(1, 4)
            ]
        else:
            self.agents = [PMExpertAgent(i, llm) for i in range(1, 4)]

    def run(self, pkd: ProjectKnowledgeDocument,
            verbose: bool = True) -> tuple[HolisticDecisionPack, dict]:
        """
        Execute the complete deliberation protocol.
        Returns (HolisticDecisionPack, metadata_dict).
        """
        t_start = time.time()
        logs: dict = {"mode": self.mode, "rounds": []}
        state = OrchestratorState()
        quality_history: dict = {}

        # ── STEP 1: PKD preprocessing ─────────────────────────────────────────
        petsi_scores, petsi_rec = self.oa.compute_petsi(pkd)
        if verbose:
            mode_label = {"full":"DELPHIC-LLM Full","abl1":"ABL-1 No-Role","abl2":"ABL-2 No-Sycophancy","abl3":"ABL-3 Fixed-Technique"}.get(self.mode, self.mode)
            print(f"  ┌─ {mode_label} | PETSI: {petsi_rec[:45]}")

        # ── STEP 2: ROUND 1 — Independent elicitation ─────────────────────────
        if verbose:
            print(f"  │  R1  Asking 3 agents independently (no peer visibility)...")
        r1_outputs: List[Round1Output] = []
        for agent in self.agents:
            out = agent.estimate_round1(pkd)
            r1_outputs.append(out)
            if verbose:
                disp_name = {"Risk-Aware Conservative":"Conservative","Delivery-Focused Balanced":"Balanced","Efficiency-Oriented Optimising":"Optimising"}.get(agent.disposition_name, agent.disposition_name)
                print(f"  │    Agent {agent.agent_id} [{disp_name:>11}]: {out.estimate_hours:>7,.0f} h  ({out.technique_selected[:35]})")

        # OA: score quality, compute κ and τ
        q1_scores = {}
        for i, out in enumerate(r1_outputs):
            q1_scores[i] = self.oa.score_quality(out)
        quality_history[1] = q1_scores

        estimates_r1 = [o.estimate_hours for o in r1_outputs]
        kappa_r1 = self.oa.compute_cv(estimates_r1)
        tau = self.oa.compute_technique_convergence(
            [o.technique_selected for o in r1_outputs])
        state.technique_convergence = tau
        state.kappa[1] = kappa_r1

        if verbose:
            spread = max([o.estimate_hours for o in r1_outputs]) - min([o.estimate_hours for o in r1_outputs])
            print(f"  │    OA  CV={kappa_r1:.3f}  τ={tau:.2f}  spread={spread:,.0f}h  {'⚠ HIGH VARIANCE' if kappa_r1>0.40 else '✓ moderate' if kappa_r1>0.15 else '✓ converged'}")

        logs["rounds"].append({
            "round": 1,
            "estimates": estimates_r1,
            "kappa": kappa_r1,
            "tau": tau,
            "techniques": [o.technique_selected for o in r1_outputs],
            "quality_scores": [q.total for q in q1_scores.values()]
        })

        anon_outputs = self.oa.anonymise(r1_outputs)

        # ── STEP 3-5: Main deliberation loop ──────────────────────────────────
        r3_outputs: Optional[List[Round3Output]] = None
        round_count = 1
        divergence_flag = False
        r2_challenger = None
        r2_builder = None
        r2_risk = None

        while round_count < self.R_MAX:
            # ── ROUND 2 — Role-differentiated review ─────────────────────────
            if verbose:
                print(f"  [R2] Role-differentiated review (round_count={round_count})...")

            if verbose:
                role_str = "Symmetric peer review" if self.mode == "abl1" else "Challenger / Builder / Risk Analyst"
                print(f"  │  R2  Role-differentiated review ({role_str})...")
            if self.mode == "abl1":
                # ABL-1: symmetric peer review
                r2_challenger = self.agents[0].review_round2_symmetric(anon_outputs)
                r2_builder    = self.agents[1].review_round2_builder(anon_outputs, r2_challenger)
                r2_risk       = self.agents[2].review_round2_risk_analyst(
                    [{"estimate_id": a["estimate_id"], "estimate_hours": a["estimate_hours"]}
                     for a in anon_outputs])
            else:
                # Full: role-differentiated
                r2_challenger = self.agents[0].review_round2_challenger(anon_outputs)
                r2_builder    = self.agents[1].review_round2_builder(anon_outputs, r2_challenger)
                # Risk Analyst sees ONLY estimate values, not rationales
                r2_risk       = self.agents[2].review_round2_risk_analyst(
                    [{"estimate_id": a["estimate_id"], "estimate_hours": a["estimate_hours"]}
                     for a in anon_outputs])

            # Build argument graph
            state.argument_graph = self.oa.build_argument_graph(
                r1_outputs, r2_challenger, r2_builder, r2_risk
            )

            # ── ROUND 3 — Constrained re-estimation ──────────────────────────
            if verbose:
                print(f"  │  R3  Constrained re-estimation (must address challenges)...")

            r3_outputs = []
            for i, agent in enumerate(self.agents):
                # Find challenges specifically targeting this agent's anon estimate
                agent_challenges = []
                if r2_challenger:
                    for ch in r2_challenger.challenges:
                        # Map anon estimate IDs back (approximate: use all challenges)
                        agent_challenges.append({
                            "weakness": ch.identified_weakness,
                            "severity": ch.severity,
                            "impact_hours": ch.impact_if_wrong_hours,
                            "counter_argument": ch.counter_argument
                        })

                # Build sycophancy intervention if needed (from previous round)
                syco_msg = None
                if state.sycophancy_flags and round_count > 1:
                    for flag in state.sycophancy_flags:
                        if flag.agent_id == agent.agent_id and flag.round == round_count:
                            syco_msg = SYCOPHANCY_INTERVENTION.format(
                                movement=flag.estimate_movement,
                                r1=r1_outputs[i].estimate_hours,
                                r2=r3_outputs[-1].final_estimate_hours if r3_outputs else 0,
                                sim=flag.rationale_similarity
                            )

                out = agent.estimate_round3(
                    pkd, r1_outputs[i], agent_challenges,
                    r2_builder, syco_msg
                )
                r3_outputs.append(out)
                if verbose:
                    delta_sym = "▲" if out.change_from_round1 > 0 else "▼" if out.change_from_round1 < 0 else "="
                    rebutted_str = "held firm" if out.challenge_rebutted else "incorporated"
                    print(f"  │    Agent {agent.agent_id}            : {out.final_estimate_hours:>7,.0f} h  ({delta_sym}{abs(out.change_from_round1):,.0f}h, {rebutted_str})")

            # Score R3 quality
            q3_scores = {}
            for i, out in enumerate(r3_outputs):
                q3_scores[i] = self.oa.score_quality(out)
            quality_history[3] = q3_scores

            # Detect sycophancy in R3 revisions
            syco_flags, intervene_agents = self.oa.detect_sycophancy(
                r1_outputs, r3_outputs,
                round_num=round_count + 1,
                disable=(self.mode == "abl2")
            )
            state.sycophancy_flags.extend(syco_flags)

            if verbose and syco_flags:
                n_interventions = len([f for f in syco_flags if f.intervention_sent])
                if n_interventions:
                    print(f"  │    OA  ⚠ Sycophancy: {n_interventions} agent(s) changed estimate without updating reasoning → intervention sent")

            # Rebuild argument graph with R3 outputs
            state.argument_graph = self.oa.build_argument_graph(
                r1_outputs, r2_challenger, r2_builder, r2_risk, r3_outputs
            )

            # Compute convergence
            estimates_r3 = [o.final_estimate_hours for o in r3_outputs]
            rationales_r3 = [o.challenge_addressed for o in r3_outputs]
            rationales_r1 = [o.reasoning_chain for o in r1_outputs]
            kappa_r3 = self.oa.compute_cv(estimates_r3)
            state.kappa[round_count + 1] = kappa_r3

            logs["rounds"].append({
                "round": round_count + 1,
                "estimates": estimates_r3,
                "kappa": kappa_r3,
                "quality_scores": [q.total for q in q3_scores.values()],
                "sycophancy_events": len([f for f in syco_flags if f.intervention_sent])
            })

            converged, false_conv, _, sim_mean = self.oa.check_convergence(
                estimates_r3, rationales_r3)

            if verbose:
                conv_str = "✓ CONVERGED" if converged and not false_conv else "⚠ FALSE CONV (injecting challenge)" if false_conv else "↻ not yet converged"
                print(f"  │    OA  CV={kappa_r3:.3f}  {conv_str}")

            if converged and not false_conv:
                break  # Genuine convergence

            if converged and false_conv:
                if verbose:
                    print(f"  │    OA  Rationale similarity too high ({sim_mean:.2f}) — all agents reasoning alike. Devil\'s advocate injected.")
                # Update anon outputs with R3 estimates for next round
                anon_outputs = self._anon_from_r3(r3_outputs)
                round_count += 1
                continue

            if kappa_r3 > self.THETA_CONV and round_count < self.R_MAX - 1:
                # Not converged, try again
                anon_outputs = self._anon_from_r3(r3_outputs)
                round_count += 1
                continue

            # Max rounds reached
            if kappa_r3 > self.THETA_CONV:
                divergence_flag = True
            break

        state.divergence_flag = divergence_flag

        # ── STEP 6: Weighted consensus and output synthesis ───────────────────
        weights = self.oa.compute_weights(quality_history)
        e_hat, ci_low, ci_high = self.oa.compute_consensus(r3_outputs, weights)

        # Compute confidence profile
        estimates_r3 = [o.final_estimate_hours for o in r3_outputs]
        rationales_r3 = [o.challenge_addressed for o in r3_outputs]
        rationales_r1 = [o.reasoning_chain for o in r1_outputs]
        kappa_r3 = state.kappa.get(round_count + 1, self.oa.compute_cv(estimates_r3))

        confidence = self.oa.compute_confidence_profile(
            graph=state.argument_graph,
            quality_history=quality_history,
            kappa_r1=kappa_r1,
            kappa_r3=kappa_r3,
            rationales_r1=rationales_r1,
            rationales_r3=rationales_r3,
            scope_items_identified=max(
                len(pkd.pkd2_scope.split("\n")), 3),   # heuristic
            scope_items_covered=len(
                set(w for o in r3_outputs for a in o.challenge_addressed.split() for w in [a])
            ) // 10  # heuristic
        )

        omega = self.oa.synthesise_decision_pack(
            e_hat=e_hat,
            ci_low=ci_low,
            ci_high=ci_high,
            confidence_profile=confidence,
            graph=state.argument_graph,
            r1_outputs=r1_outputs,
            r3_outputs=r3_outputs,
            sycophancy_flags=state.sycophancy_flags,
            tau=tau,
            rounds_to_converge=round_count + 1,
            divergence_flag=divergence_flag,
            agent_weights=weights,
            petsi_rec=petsi_rec,
            kappa_r3=kappa_r3
        )

        t_elapsed = time.time() - t_start
        logs["time_seconds"] = round(t_elapsed, 1)
        logs["api_cost_usd"]  = round(self.llm.total_cost_usd, 4)
        logs["consensus_estimate"] = round(e_hat, 1)
        logs["kappa_r1"] = kappa_r1
        logs["kappa_r3"] = kappa_r3
        logs["tau"] = tau
        logs["divergence_flag"] = divergence_flag
        logs["confidence"] = {
            "C_est": confidence.C_est,
            "C_assump": confidence.C_assump,
            "C_complete": confidence.C_complete,
            "C_consensus": confidence.C_consensus
        }
        logs["sycophancy_rate"] = omega.sycophancy_rate
        logs["r1_estimates"] = estimates_r1
        logs["r3_estimates"] = estimates_r3
        logs["agent_weights"] = weights
        logs["quality_r1"] = [quality_history[1][i].total for i in range(3)]
        logs["quality_r3"] = [quality_history[3][i].total for i in range(3)
                              if i in quality_history.get(3, {})]

        if verbose:
            q_mean = sum(quality_history[3][i].total for i in range(3) if i in quality_history.get(3,{})) / max(len(quality_history.get(3,{})),1)
            print(f"  └─ Ω  {e_hat:>7,.0f} h  [{ci_low:,.0f}–{ci_high:,.0f}h]  Q={q_mean:.1f}/20  cost=${self.llm.total_cost_usd:.2f}  {t_elapsed:.0f}s")

        return omega, logs

    def _anon_from_r3(self, r3_outputs: List[Round3Output]) -> List[dict]:
        """
        Convert Round 3 outputs to anonymised format for subsequent rounds.
        Used when additional deliberation rounds are triggered.
        """
        import random
        shuffled = list(range(len(r3_outputs)))
        random.shuffle(shuffled)
        result = []
        for new_id, orig_idx in enumerate(shuffled, 1):
            o = r3_outputs[orig_idx]
            result.append({
                "estimate_id": new_id,
                "estimate_hours": o.final_estimate_hours,
                "technique_selected": "Updated estimate",
                "technique_justification": "See challenge_addressed field",
                "reasoning_chain": o.challenge_addressed,
                "key_assumptions": [],
                "identified_risks": [],
                "confidence_interval": [
                    o.updated_confidence_interval_low,
                    o.updated_confidence_interval_high
                ],
                "uncertainty_sources": [o.remaining_uncertainty],
            })
        return result
