"""
DELPHIC-LLM: PM-Expert Agent
Implements Ψᵢ = (K, Dᵢ) — shared knowledge base K + distinct disposition Dᵢ.
Each agent independently selects estimation techniques.
"""
from typing import List, Optional

from delphic_llm.models import (
    Round1Output, Round2ChallengerOutput, Round2BuilderOutput,
    Round2RiskAnalystOutput, Round3Output, ProjectKnowledgeDocument
)
from delphic_llm.llm_client import LLMClient
from delphic_llm.prompts.templates import (
    PM_KNOWLEDGE_BASE, DISPOSITION_D1, DISPOSITION_D2, DISPOSITION_D3,
    ROUND1_TASK, ROUND2_CHALLENGER_TASK, ROUND2_BUILDER_TASK,
    ROUND2_RISK_ANALYST_TASK, ROUND3_TASK, SYCOPHANCY_INTERVENTION
)
import json


DISPOSITIONS = {1: DISPOSITION_D1, 2: DISPOSITION_D2, 3: DISPOSITION_D3}
DISPOSITION_NAMES = {
    1: "Risk-Aware Conservative",
    2: "Delivery-Focused Balanced",
    3: "Efficiency-Oriented Optimising"
}

# Fixed technique assignments for ABL-3
ABL3_TECHNIQUES = {
    1: "Three-point PERT estimation (E=(O+4M+P)/6, PMBOK §6.4.2.3)",
    2: "Parametric modelling using unit rates",
    3: "Analogy-based estimation using historical project similarity"
}


class PMExpertAgent:
    """
    A fully qualified PM expert agent.
    Ψᵢ = (K, Dᵢ): shared PM knowledge base K, distinct reasoning disposition Dᵢ.
    """

    def __init__(self, agent_id: int, llm: LLMClient,
                 fixed_technique: Optional[str] = None):
        """
        agent_id: 1, 2, or 3 (determines disposition)
        fixed_technique: for ABL-3 ablation only
        """
        assert agent_id in (1, 2, 3), "agent_id must be 1, 2, or 3"
        self.agent_id = agent_id
        self.llm = llm
        self.fixed_technique = fixed_technique
        self.disposition_name = DISPOSITION_NAMES[agent_id]

        # Build system prompt = K + Dᵢ
        self.system_prompt = PM_KNOWLEDGE_BASE + "\n" + DISPOSITIONS[agent_id]
        if fixed_technique:
            self.system_prompt += (
                f"\n\nFIXED TECHNIQUE CONSTRAINT (ABL-3 ablation): "
                f"You MUST use {fixed_technique} for this estimation. "
                f"Do not select any other technique."
            )

    def estimate_round1(self, pkd: ProjectKnowledgeDocument) -> Round1Output:
        """Round 1: independent elicitation from isolated C_PKD context."""
        user_prompt = ROUND1_TASK.format(pkd=pkd.to_context_string())
        return self.llm.agent_call(
            system=self.system_prompt,
            user=user_prompt,
            schema=Round1Output
        )

    def review_round2_challenger(self,
                                  anonymised_estimates: List[dict]) -> Round2ChallengerOutput:
        """Round 2: pure adversarial critique — Challenger role."""
        estimates_str = json.dumps(anonymised_estimates, indent=2)
        user_prompt = ROUND2_CHALLENGER_TASK.format(
            anonymised_estimates=estimates_str
        )
        return self.llm.agent_call(
            system=self.system_prompt,
            user=user_prompt,
            schema=Round2ChallengerOutput
        )

    def review_round2_builder(self,
                               anonymised_estimates: List[dict],
                               challenger_report: Round2ChallengerOutput
                               ) -> Round2BuilderOutput:
        """Round 2: constructive synthesis — Builder role."""
        estimates_str = json.dumps(anonymised_estimates, indent=2)
        challenger_str = challenger_report.model_dump_json(indent=2)
        user_prompt = ROUND2_BUILDER_TASK.format(
            anonymised_estimates=estimates_str,
            challenger_report=challenger_str
        )
        return self.llm.agent_call(
            system=self.system_prompt,
            user=user_prompt,
            schema=Round2BuilderOutput
        )

    def review_round2_risk_analyst(self,
                                    estimate_values: List[dict]) -> Round2RiskAnalystOutput:
        """Round 2: independent risk identification — Risk Analyst role.
        Receives ONLY estimate values, NOT rationales."""
        estimates_str = json.dumps(estimate_values, indent=2)
        user_prompt = ROUND2_RISK_ANALYST_TASK.format(
            anonymised_estimate_values=estimates_str
        )
        return self.llm.agent_call(
            system=self.system_prompt,
            user=user_prompt,
            schema=Round2RiskAnalystOutput
        )

    def review_round2_symmetric(self,
                                  anonymised_estimates: List[dict]) -> Round2ChallengerOutput:
        """
        ABL-1 ablation: symmetric peer review (no role differentiation).
        All agents use the same undifferentiated review prompt.
        """
        estimates_str = json.dumps(anonymised_estimates, indent=2)
        symmetric_prompt = f"""You have been asked to review three anonymised project estimates.

THREE ANONYMISED ESTIMATES:
{estimates_str}

Please review all three estimates. For each estimate:
1. Identify any weaknesses or questionable assumptions
2. Rate the severity (HIGH/MEDIUM/LOW)
3. Suggest what counter-evidence might apply
4. Estimate the impact if the assumption is wrong

Respond ONLY with valid JSON matching the Round2ChallengerOutput schema."""
        return self.llm.agent_call(
            system=self.system_prompt,
            user=symmetric_prompt,
            schema=Round2ChallengerOutput
        )

    def estimate_round3(self,
                         pkd: ProjectKnowledgeDocument,
                         r1_output: Round1Output,
                         challenges_against: list,
                         builder_output: Round2BuilderOutput,
                         sycophancy_intervention: Optional[str] = None
                         ) -> Round3Output:
        """Round 3: constrained informed re-estimation."""
        challenges_str = json.dumps(challenges_against, indent=2)
        builder_str = (f"Synthesis proposal: {builder_output.proposed_synthesis_hours:.0f} hours\n"
                       f"Rationale: {builder_output.synthesis_rationale}\n"
                       f"Unresolved tensions: {'; '.join(builder_output.unresolved_tensions)}")

        user_prompt = ROUND3_TASK.format(
            pkd=pkd.to_context_string(),
            r1_estimate=f"{r1_output.estimate_hours:.0f}",
            r1_technique=r1_output.technique_selected,
            challenges_against_you=challenges_str,
            builder_synthesis=builder_str
        )

        if sycophancy_intervention:
            user_prompt = sycophancy_intervention + "\n\n" + user_prompt

        return self.llm.agent_call(
            system=self.system_prompt,
            user=user_prompt,
            schema=Round3Output
        )
