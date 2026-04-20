"""
DELPHIC-LLM: Orchestrator Agent (OA)
Implements the complete meta-cognitive governance layer:
  - Argument quality scoring (5-criterion rubric)
  - Argument graph construction
  - Sycophancy detection (two-stage)
  - Convergence detection (numerical + false-convergence guard)
  - Confidence profile synthesis
  - Holistic Decision Pack generation
  - PETSI computation
"""
import json
import random
import re
from statistics import mean, stdev, median
from typing import List, Optional, Tuple, Dict
import numpy as np

from delphic_llm.models import (
    Round1Output, Round2ChallengerOutput, Round2BuilderOutput,
    Round2RiskAnalystOutput, Round3Output,
    ArgumentGraph, ArgumentNode, ArgumentEdge,
    QualityScores, SycophancyFlag, OrchestratorState,
    ConfidenceProfile, ScenarioEnvelope, AssumptionRegister,
    HolisticDecisionPack, ProjectKnowledgeDocument
)
from delphic_llm.llm_client import LLMClient
from delphic_llm.prompts.templates import QUALITY_SCORING_PROMPT

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False


class Orchestrator:
    """
    Meta-cognitive governance agent.
    Does NOT produce effort estimates — governs the deliberation process.
    """

    # Thresholds from paper
    THETA_CONV   = 0.15   # convergence threshold on CV
    THETA_HIGH   = 0.40   # high-variance threshold
    THETA_NORM   = 0.50   # normalisation factor for C_est
    SYCO_MOVE    = 0.10   # minimum estimate movement to investigate (10%)
    SYCO_SIM     = 0.90   # rationale similarity threshold for sycophancy
    FALSE_CONV   = 0.95   # mean pairwise rationale similarity → false convergence
    ROUND_WEIGHTS = (0.20, 0.50, 0.30)  # α₁, α₂, α₃

    def __init__(self, llm: LLMClient, embed_model_name: str = "all-mpnet-base-v2"):
        self.llm = llm
        self._embed_model: Optional[object] = None
        self._embed_model_name = embed_model_name

    @property
    def embed_model(self):
        if self._embed_model is None:
            if not ST_AVAILABLE:
                raise ImportError(
                    "sentence-transformers required: pip install sentence-transformers")
            # Check module-level cache first so model only loads once per process
            cache_key = f"_st_model_{self._embed_model_name}"
            import delphic_llm.orchestrator as _orch_mod
            if not hasattr(_orch_mod, cache_key):
                print(f"  [OA] Loading embedding model {self._embed_model_name} (once)...")
                setattr(_orch_mod, cache_key, SentenceTransformer(self._embed_model_name))
            self._embed_model = getattr(_orch_mod, cache_key)
        return self._embed_model

    # ── Quality scoring ───────────────────────────────────────────────────────
    def score_quality(self, output: Round1Output | Round3Output) -> QualityScores:
        """Score agent output against 5-criterion rubric Q (0-4 each, max 20)."""
        output_str = output.model_dump_json(indent=2)
        prompt = QUALITY_SCORING_PROMPT.format(agent_output=output_str)
        result = self.llm.orchestrator_call(
            system="You are an expert evaluator of project estimation reasoning quality.",
            user=prompt
        )
        if isinstance(result, dict):
            return QualityScores(
                q1_technique_justification=float(result.get("q1", 2)),
                q2_assumption_explicitness=float(result.get("q2", 2)),
                q3_reasoning_traceability=float(result.get("q3", 2)),
                q4_risk_identification=float(result.get("q4", 2)),
                q5_uncertainty_acknowledgement=float(result.get("q5", 2))
            )
        # Fallback: parse scores from text
        return self._parse_quality_scores_from_text(str(result))

    def _parse_quality_scores_from_text(self, text: str) -> QualityScores:
        scores = []
        for key in ["q1", "q2", "q3", "q4", "q5"]:
            m = re.search(rf'"{key}":\s*(\d+(?:\.\d+)?)', text)
            scores.append(float(m.group(1)) if m else 2.0)
        return QualityScores(
            q1_technique_justification=scores[0],
            q2_assumption_explicitness=scores[1],
            q3_reasoning_traceability=scores[2],
            q4_risk_identification=scores[3],
            q5_uncertainty_acknowledgement=scores[4]
        )

    # ── Anonymisation ─────────────────────────────────────────────────────────
    def anonymise(self, outputs: List[Round1Output]) -> List[dict]:
        """Remove any agent identity markers, shuffle randomly, assign IDs 1-3."""
        shuffled = list(outputs)
        random.shuffle(shuffled)
        result = []
        for i, o in enumerate(shuffled, 1):
            result.append({
                "estimate_id": i,
                "estimate_hours": o.estimate_hours,
                "technique_selected": o.technique_selected,
                "technique_justification": o.technique_justification,
                "reasoning_chain": o.reasoning_chain,
                "key_assumptions": o.key_assumptions,
                "identified_risks": o.identified_risks,
                "confidence_interval": [o.confidence_interval_low, o.confidence_interval_high],
                "uncertainty_sources": o.uncertainty_sources,
            })
        return result

    # ── Convergence coefficient ────────────────────────────────────────────────
    def compute_cv(self, estimates: List[float]) -> float:
        """κ = CV = std / mean of estimates."""
        if len(estimates) < 2:
            return 0.0
        m = mean(estimates)
        if m == 0:
            return 0.0
        return stdev(estimates) / m

    # ── Technique convergence ─────────────────────────────────────────────────
    def compute_technique_convergence(self, techniques: List[str]) -> float:
        """τ⁽¹⁾ = |{i: Tᵢ = mode(T)}| / n"""
        if not techniques:
            return 0.0
        # Normalise technique names for comparison
        normalised = [t.lower().strip().split("(")[0].strip() for t in techniques]
        from collections import Counter
        most_common_count = Counter(normalised).most_common(1)[0][1]
        return most_common_count / len(techniques)

    # ── Argument graph construction ───────────────────────────────────────────
    def build_argument_graph(self,
                              r1_outputs: List[Round1Output],
                              r2_challenger: Round2ChallengerOutput,
                              r2_builder: Round2BuilderOutput,
                              r2_risk: Round2RiskAnalystOutput,
                              r3_outputs: Optional[List[Round3Output]] = None
                              ) -> ArgumentGraph:
        """Construct G = (V, E) from all deliberation artefacts."""
        graph = ArgumentGraph()
        node_counter = [0]

        def add_node(ntype, content, agent_id=None, round_num=None) -> str:
            node_counter[0] += 1
            nid = f"N{node_counter[0]:03d}"
            graph.nodes.append(ArgumentNode(
                node_id=nid, node_type=ntype, content=content,
                agent_id=agent_id, round=round_num
            ))
            return nid

        # R1: add assumptions as nodes
        r1_assumption_nodes: Dict[int, List[str]] = {}
        for i, out in enumerate(r1_outputs, 1):
            agent_nodes = []
            for assumption in out.key_assumptions:
                nid = add_node("ASSUMPTION", assumption, agent_id=i, round_num=1)
                agent_nodes.append(nid)
            r1_assumption_nodes[i] = agent_nodes
            # Add claim node for the estimate itself
            add_node("CLAIM",
                     f"Agent {i} estimates {out.estimate_hours:.0f} hours using {out.technique_selected}",
                     agent_id=i, round_num=1)

        # R2 Challenger: add ATTACKS edges
        for challenge in r2_challenger.challenges:
            weakness_nid = add_node("CHALLENGE",
                                    challenge.identified_weakness,
                                    round_num=2)
            # Try to link to the matching assumption node
            target_nodes = r1_assumption_nodes.get(challenge.target_estimate_id, [])
            # Find closest matching assumption
            best_target = self._find_closest_node(
                challenge.identified_weakness,
                [n for n in graph.nodes if n.node_type == "ASSUMPTION" and
                 n.agent_id == challenge.target_estimate_id]
            )
            if best_target:
                graph.edges.append(ArgumentEdge(
                    source_id=weakness_nid, target_id=best_target,
                    edge_type="ATTACKS",
                    weight=1.0 if challenge.severity == "HIGH" else
                           0.6 if challenge.severity == "MEDIUM" else 0.3
                ))

        # R2 Builder: add SYNTHESISES and SUPPORTS edges
        synth_nid = add_node("SYNTHESIS",
                             r2_builder.synthesis_rationale[:200],
                             round_num=2)
        for elem in r2_builder.supported_elements:
            support_nid = add_node("SUPPORT", elem.element, round_num=2)
            graph.edges.append(ArgumentEdge(
                source_id=support_nid, target_id=synth_nid,
                edge_type="SYNTHESISES",
                weight=1.0 if elem.confidence == "HIGH" else
                       0.6 if elem.confidence == "MEDIUM" else 0.3
            ))

        # R3: add REBUTS edges where challenge_rebutted=True
        if r3_outputs:
            for i, out in enumerate(r3_outputs, 1):
                if out.challenge_rebutted:
                    rebuttal_nid = add_node("CLAIM",
                                           f"Agent {i} rebuts: {out.challenge_addressed[:150]}",
                                           agent_id=i, round_num=3)
                    # Find challenge nodes targeting this agent
                    challenge_nodes = [
                        n for n in graph.nodes
                        if n.node_type == "CHALLENGE" and n.round == 2
                    ]
                    if challenge_nodes:
                        graph.edges.append(ArgumentEdge(
                            source_id=rebuttal_nid,
                            target_id=challenge_nodes[0].node_id,
                            edge_type="REBUTS",
                            weight=0.8
                        ))

        return graph

    def _find_closest_node(self, text: str, nodes: List[ArgumentNode]) -> Optional[str]:
        """Simple heuristic: find node whose content has most word overlap."""
        if not nodes:
            return None
        text_words = set(text.lower().split())
        best_id, best_score = nodes[0].node_id, 0
        for n in nodes:
            overlap = len(text_words & set(n.content.lower().split()))
            if overlap > best_score:
                best_score = overlap
                best_id = n.node_id
        return best_id

    # ── Sycophancy detection ──────────────────────────────────────────────────
    def detect_sycophancy(self,
                          r_prev: List[Round1Output] | List[Round3Output],
                          r_curr: List[Round3Output],
                          round_num: int,
                          disable: bool = False
                          ) -> Tuple[List[SycophancyFlag], List[int]]:
        """
        Two-stage sycophancy detection.
        Returns (flags, agent_ids_to_intervene).
        disable=True for ABL-2 ablation.
        """
        if disable:
            return [], []

        flags = []
        intervene_agents = []

        prev_estimates = [o.estimate_hours if isinstance(o, Round1Output)
                         else o.final_estimate_hours for o in r_prev]
        prev_rationales = [o.reasoning_chain if isinstance(o, Round1Output)
                          else o.challenge_addressed for o in r_prev]
        curr_estimates  = [o.final_estimate_hours for o in r_curr]
        curr_rationales = [o.challenge_addressed for o in r_curr]

        for i, (pe, ce, pr, cr) in enumerate(
            zip(prev_estimates, curr_estimates, prev_rationales, curr_rationales)):
            if pe == 0:
                continue
            delta = abs(ce - pe) / pe   # |δᵢ|

            # Stage 1: only investigate if estimate moved > 10%
            if delta <= self.SYCO_MOVE:
                continue

            # Stage 2: compute rationale similarity
            sim = self._compute_embedding_similarity(pr, cr)
            flag = SycophancyFlag(
                agent_id=i + 1,
                round=round_num,
                estimate_movement=delta,
                rationale_similarity=sim
            )
            flags.append(flag)

            if sim > self.SYCO_SIM:
                # Estimate changed but reasoning didn't → sycophancy
                flag.intervention_sent = True
                intervene_agents.append(i)

        return flags, intervene_agents

    def _compute_embedding_similarity(self, text1: str, text2: str) -> float:
        """Cosine similarity between sentence embeddings."""
        try:
            embeddings = self.embed_model.encode([text1, text2], normalize_embeddings=True)
            return float(np.dot(embeddings[0], embeddings[1]))
        except Exception:
            # Fallback: Jaccard similarity if embedding fails
            w1 = set(text1.lower().split())
            w2 = set(text2.lower().split())
            if not w1 or not w2:
                return 0.0
            return len(w1 & w2) / len(w1 | w2)

    def compute_mean_pairwise_similarity(self, rationales: List[str]) -> float:
        """Mean pairwise cosine similarity for false-convergence guard."""
        if len(rationales) < 2:
            return 0.0
        similarities = []
        for i in range(len(rationales)):
            for j in range(i + 1, len(rationales)):
                sim = self._compute_embedding_similarity(rationales[i], rationales[j])
                similarities.append(sim)
        return mean(similarities) if similarities else 0.0

    # ── Convergence detection ─────────────────────────────────────────────────
    def check_convergence(self,
                           estimates: List[float],
                           rationales: List[str]) -> Tuple[bool, bool, float, float]:
        """
        Returns (numerical_converged, false_convergence_detected, kappa, sim_mean).
        false_convergence_detected=True → inject devil's advocate.
        """
        kappa = self.compute_cv(estimates)
        numerical = kappa <= self.THETA_CONV
        if not numerical:
            return False, False, kappa, 0.0
        sim_mean = self.compute_mean_pairwise_similarity(rationales)
        false_conv = sim_mean > self.FALSE_CONV
        return True, false_conv, kappa, sim_mean

    # ── Weighted consensus ────────────────────────────────────────────────────
    def compute_weights(self, quality_history: dict) -> List[float]:
        """
        wᵢ = Σᵣ αᵣ · qᵢ⁽ʳ⁾  where α=(0.20, 0.50, 0.30)
        quality_history: {round_num: {agent_idx: QualityScores}}
        Returns normalised weights for 3 agents.
        """
        n_agents = 3
        raw_weights = [0.0] * n_agents
        for r_idx, alpha in enumerate(self.ROUND_WEIGHTS, 1):
            round_scores = quality_history.get(r_idx, {})
            for a_idx in range(n_agents):
                q = round_scores.get(a_idx)
                if q is not None:
                    raw_weights[a_idx] += alpha * q.total
                else:
                    raw_weights[a_idx] += alpha * 10.0  # default 50% score

        total = sum(raw_weights)
        if total == 0:
            return [1/n_agents] * n_agents
        return [w / total for w in raw_weights]

    def compute_consensus(self,
                           r3_outputs: List[Round3Output],
                           weights: List[float]) -> Tuple[float, float, float]:
        """
        Returns (Ê, CI_low, CI_high).
        Ê = weighted mean; CI reflects spread of weighted agent estimates.
        """
        estimates = [o.final_estimate_hours for o in r3_outputs]
        e_hat = sum(w * e for w, e in zip(weights, estimates))

        # CI: weighted spread of agent CIs
        ci_lows  = [o.updated_confidence_interval_low  for o in r3_outputs]
        ci_highs = [o.updated_confidence_interval_high for o in r3_outputs]
        ci_low  = sum(w * l for w, l in zip(weights, ci_lows))
        ci_high = sum(w * h for w, h in zip(weights, ci_highs))
        return e_hat, ci_low, ci_high

    # ── Confidence synthesis ──────────────────────────────────────────────────
    def compute_confidence_profile(self,
                                    graph: ArgumentGraph,
                                    quality_history: dict,
                                    kappa_r1: float,
                                    kappa_r3: float,
                                    rationales_r1: List[str],
                                    rationales_r3: List[str],
                                    scope_items_identified: int,
                                    scope_items_covered: int
                                    ) -> ConfidenceProfile:
        """Compute all four confidence axes from deliberation dynamics."""

        # C_est: estimation stability × reasoning quality
        mean_q3 = mean([qs.total for qs in quality_history.get(3, {}).values()]
                       if quality_history.get(3) else [10.0])
        c_est = max(0.0, min(1.0,
            (1 - min(kappa_r3 / self.THETA_NORM, 1.0)) * (mean_q3 / 20.0)
        ))

        # C_assump: fraction of assumptions NOT contested
        contested = graph.get_contested_assumptions()
        total_assumptions = len([n for n in graph.nodes if n.node_type == "ASSUMPTION"])
        if total_assumptions > 0:
            c_assump = max(0.0, 1 - len(contested) / total_assumptions)
        else:
            c_assump = 0.5

        # C_complete: scope coverage
        if scope_items_identified > 0:
            c_complete = min(1.0, scope_items_covered / scope_items_identified)
        else:
            c_complete = 0.6  # default when PKD4/5 absent (NASA93 case)

        # C_consensus: congruence of estimate and rationale convergence
        sim_est_conv  = max(0.0, 1 - kappa_r3 / max(kappa_r1, 0.001))
        sim_rat_conv  = self.compute_mean_pairwise_similarity(rationales_r3) - \
                        self.compute_mean_pairwise_similarity(rationales_r1)
        sim_rat_conv  = max(0.0, min(1.0, sim_rat_conv + 0.5))  # normalise to [0,1]
        c_consensus   = max(0.0, min(1.0, 1 - abs(sim_est_conv - sim_rat_conv)))

        return ConfidenceProfile(
            C_est=round(c_est, 3),
            C_assump=round(c_assump, 3),
            C_complete=round(c_complete, 3),
            C_consensus=round(c_consensus, 3)
        )

    # ── PETSI ─────────────────────────────────────────────────────────────────
    def compute_petsi(self, pkd: ProjectKnowledgeDocument) -> Tuple[dict, str]:
        """
        Compute PETSI scores from PKD and return (scores_dict, recommended_technique).
        Scores are heuristic proxies derived from PKD completeness.
        """
        # Heuristic scoring from PKD field presence and content length
        def completeness(field: str) -> float:
            return min(1.0, len(field.strip()) / 500) if field.strip() else 0.0

        S_c = completeness(pkd.pkd2_scope)           # scope clarity
        H_a = completeness(pkd.pkd5_analogues)        # historical data
        T_e = completeness(pkd.pkd3_resources) * 0.7  # team experience proxy
        N_t = 1.0 - completeness(pkd.pkd5_analogues)  # novelty = 1 - history
        I_c = mean([completeness(f) for f in [
            pkd.pkd1_context, pkd.pkd2_scope, pkd.pkd3_resources
        ]])

        scores = {"S_c": S_c, "H_a": H_a, "T_e": T_e, "N_t": N_t, "I_c": I_c}

        # Recommend technique
        if S_c > 0.70 and I_c > 0.60:
            rec = "Bottom-up WBS estimation"
        elif H_a > 0.70 and T_e > 0.60:
            rec = "Parametric modelling"
        elif H_a > 0.50 and N_t < 0.50:
            rec = "Analogy-based estimation"
        elif S_c < 0.70:
            rec = "Three-point PERT"
        else:
            rec = "DELPHIC-LLM (hybrid deliberation)"

        return scores, rec

    # ── Decision Pack synthesis ───────────────────────────────────────────────
    def synthesise_decision_pack(self,
                                  e_hat: float,
                                  ci_low: float,
                                  ci_high: float,
                                  confidence_profile: ConfidenceProfile,
                                  graph: ArgumentGraph,
                                  r1_outputs: List[Round1Output],
                                  r3_outputs: List[Round3Output],
                                  sycophancy_flags: List[SycophancyFlag],
                                  tau: float,
                                  rounds_to_converge: int,
                                  divergence_flag: bool,
                                  agent_weights: List[float],
                                  petsi_rec: str,
                                  kappa_r3: float
                                  ) -> HolisticDecisionPack:
        """Build the complete Holistic Intelligent Decision Pack Ω."""

        # Ω₃: scenario envelope from agent dispositions
        sorted_estimates = sorted([o.final_estimate_hours for o in r3_outputs])
        s_opt  = sorted_estimates[0]   # efficiency-oriented agent
        s_pess = sorted_estimates[-1]  # risk-aware agent
        opt_conditions  = "All identified risks do not materialise; team achieves modelled efficiency"
        pess_conditions = "All contested assumptions fail in most adverse direction; " + \
                         "; ".join(graph.get_contested_assumptions()[:2]) if \
                         graph.get_contested_assumptions() else \
                         "Scope expands beyond current definition"

        scenario = ScenarioEnvelope(
            optimistic_hours=s_opt,
            base_hours=e_hat,
            pessimistic_hours=s_pess,
            optimistic_conditions=opt_conditions,
            pessimistic_conditions=pess_conditions
        )

        # Ω₄: R/A/G assumption register
        contested = graph.get_contested_assumptions()
        supported = graph.get_supported_assumptions()
        all_assumptions = [a for out in r1_outputs for a in out.key_assumptions]
        amber = [a for a in all_assumptions
                 if a not in contested and a not in supported]

        register = AssumptionRegister(
            red_assumptions=contested[:5],
            amber_assumptions=amber[:5],
            green_assumptions=supported[:5]
        )

        # Ω₅: actionable recommendations
        recommendations = []
        cp = confidence_profile

        if cp.C_assump < 0.40:
            recommendations.append(
                f"CRITICAL: Resolve {len(contested)} Red-flagged assumption(s) before committing "
                f"to this estimate (C_assump={cp.C_assump:.2f})."
            )
        if cp.C_complete < 0.60:
            recommendations.append(
                f"Decompose scope further before estimating — completeness confidence is low "
                f"(C_complete={cp.C_complete:.2f})."
            )
        if cp.C_consensus < 0.50:
            recommendations.append(
                f"Deliberation was inconclusive (C_consensus={cp.C_consensus:.2f}). "
                f"Consider human expert Delphi for higher-stakes commitment."
            )
        if tau <= 0.34:
            recommendations.append(
                f"Agents selected different estimation techniques (τ={tau:.2f}), indicating "
                f"scope ambiguity. Clarify project definition before re-estimating."
            )
        if kappa_r3 > 0.15:
            recommendations.append(
                f"Estimate spread remains high (CV={kappa_r3:.2f}). "
                f"Range: {s_opt:.0f}–{s_pess:.0f} hours. "
                f"Use the pessimistic scenario for fixed-price commitments."
            )
        if petsi_rec and "DELPHIC" not in petsi_rec:
            recommendations.append(
                f"PETSI suggests {petsi_rec} may be sufficient for this project type. "
                f"Consider using it as a complementary check."
            )

        # Compute sycophancy rate
        sr = len(sycophancy_flags) / max(len(r3_outputs) * 2, 1)

        return HolisticDecisionPack(
            consensus_estimate_hours=round(e_hat, 1),
            ci_low_hours=round(ci_low, 1),
            ci_high_hours=round(ci_high, 1),
            confidence_profile=confidence_profile,
            scenario_envelope=scenario,
            assumption_register=register,
            recommendations=recommendations[:5],
            technique_convergence=round(tau, 3),
            sycophancy_rate=round(sr, 3),
            rounds_to_converge=rounds_to_converge,
            divergence_flag=divergence_flag,
            agent_weights=[round(w, 3) for w in agent_weights],
            petsi_recommended_technique=petsi_rec
        )
