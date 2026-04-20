"""
DELPHIC-LLM: Data models
All Pydantic schemas matching the paper's formal specification exactly.
"""
from typing import List, Optional, Literal
from pydantic import BaseModel, Field


# ─── Round 1: Independent elicitation output ──────────────────────────────────
class Round1Output(BaseModel):
    technique_selected: str = Field(
        description="Chosen estimation technique(s) with PMBOK/standard reference"
    )
    technique_justification: str = Field(
        description="Why this technique fits this specific project"
    )
    estimate_hours: float = Field(
        description="Point estimate in person-hours", gt=0
    )
    confidence_interval_low: float = Field(
        description="Lower bound of 80% CI in person-hours", gt=0
    )
    confidence_interval_high: float = Field(
        description="Upper bound of 80% CI in person-hours", gt=0
    )
    key_assumptions: List[str] = Field(
        description="Up to 5 falsifiable assumptions",
        max_length=5
    )
    identified_risks: List[str] = Field(
        description="Up to 5 risks with estimated impact in hours",
        max_length=5
    )
    reasoning_chain: str = Field(
        description="Step-by-step derivation linking project inputs to estimate"
    )
    uncertainty_sources: List[str] = Field(
        description="Explicit drivers of confidence interval width"
    )


# ─── Round 2: Role outputs ────────────────────────────────────────────────────
class ChallengerChallenge(BaseModel):
    target_estimate_id: int = Field(description="Which anonymised estimate (1/2/3)")
    identified_weakness: str = Field(description="The specific assumption being challenged")
    severity: Literal["HIGH", "MEDIUM", "LOW"]
    counter_argument: str = Field(description="Evidence or reasoning against the assumption")
    impact_if_wrong_hours: float = Field(
        description="Estimated additional error in hours if assumption fails"
    )


class Round2ChallengerOutput(BaseModel):
    challenges: List[ChallengerChallenge] = Field(
        description="One challenge per estimate (3 total)"
    )
    overall_assessment: str = Field(
        description="Brief summary of deliberation weaknesses identified"
    )


class SupportedElement(BaseModel):
    estimate_id: int
    element: str
    confidence: Literal["HIGH", "MEDIUM", "LOW"]


class Round2BuilderOutput(BaseModel):
    supported_elements: List[SupportedElement]
    genuine_agreement_areas: List[str] = Field(
        description="Areas where all estimates agree substantively"
    )
    proposed_synthesis_hours: float = Field(gt=0)
    synthesis_rationale: str
    unresolved_tensions: List[str] = Field(
        description="MANDATORY: what the synthesis cannot reconcile"
    )


class RiskItem(BaseModel):
    estimate_id: int
    critical_assumption: str
    failure_scenario: str
    probability: Literal["HIGH", "MEDIUM", "LOW"]
    impact_hours: float
    hedge_recommendation: str


class Round2RiskAnalystOutput(BaseModel):
    risk_items: List[RiskItem] = Field(description="One risk item per estimate")
    overall_risk_level: Literal["HIGH", "MEDIUM", "LOW"]


# ─── Round 3: Constrained re-estimation ──────────────────────────────────────
class Round3Output(BaseModel):
    final_estimate_hours: float = Field(gt=0)
    change_from_round1: float = Field(
        description="Signed difference from Round 1 (positive = increased)"
    )
    challenge_addressed: str = Field(
        description="How the agent responded to the top challenge against its Round 1 estimate"
    )
    challenge_rebutted: bool = Field(
        description="True if agent maintained original position; False if incorporated"
    )
    updated_confidence_interval_low: float = Field(gt=0)
    updated_confidence_interval_high: float = Field(gt=0)
    remaining_uncertainty: str = Field(
        description="One sentence: what is still unknown"
    )
    technique_maintained: bool = Field(
        description="True if same technique as Round 1"
    )
    technique_switch_reason: Optional[str] = Field(
        default=None,
        description="Required if technique_maintained=False"
    )


# ─── Orchestrator state ───────────────────────────────────────────────────────
class ArgumentNode(BaseModel):
    node_id: str
    node_type: Literal["ASSUMPTION", "CLAIM", "RISK", "SYNTHESIS", "CHALLENGE", "SUPPORT"]
    content: str
    agent_id: Optional[int] = None
    round: Optional[int] = None


class ArgumentEdge(BaseModel):
    source_id: str
    target_id: str
    edge_type: Literal["ATTACKS", "SUPPORTS", "REBUTS", "SYNTHESISES"]
    weight: float = Field(default=1.0, ge=0.0, le=1.0)


class ArgumentGraph(BaseModel):
    nodes: List[ArgumentNode] = Field(default_factory=list)
    edges: List[ArgumentEdge] = Field(default_factory=list)

    def get_contested_assumptions(self) -> List[str]:
        """Return assumptions that have ATTACK edges but no REBUTTAL edges."""
        attacked = {e.target_id for e in self.edges if e.edge_type == "ATTACKS"}
        rebutted = {e.target_id for e in self.edges if e.edge_type == "REBUTS"}
        contested_ids = attacked - rebutted
        return [n.content for n in self.nodes
                if n.node_id in contested_ids and n.node_type == "ASSUMPTION"]

    def get_supported_assumptions(self) -> List[str]:
        """Return assumptions with SUPPORT edges from multiple agents."""
        support_counts: dict = {}
        for e in self.edges:
            if e.edge_type == "SUPPORTS":
                support_counts[e.target_id] = support_counts.get(e.target_id, 0) + 1
        supported_ids = {nid for nid, cnt in support_counts.items() if cnt >= 2}
        return [n.content for n in self.nodes
                if n.node_id in supported_ids and n.node_type == "ASSUMPTION"]


class SycophancyFlag(BaseModel):
    agent_id: int
    round: int
    estimate_movement: float   # δᵢ
    rationale_similarity: float  # cosine sim
    intervention_sent: bool = False


class QualityScores(BaseModel):
    """Five-criterion rubric, 0-4 each, max 20."""
    q1_technique_justification: float = Field(ge=0, le=4)
    q2_assumption_explicitness: float = Field(ge=0, le=4)
    q3_reasoning_traceability: float = Field(ge=0, le=4)
    q4_risk_identification: float = Field(ge=0, le=4)
    q5_uncertainty_acknowledgement: float = Field(ge=0, le=4)

    @property
    def total(self) -> float:
        return (self.q1_technique_justification + self.q2_assumption_explicitness +
                self.q3_reasoning_traceability + self.q4_risk_identification +
                self.q5_uncertainty_acknowledgement)


class OrchestratorState(BaseModel):
    round: int = 0
    argument_graph: ArgumentGraph = Field(default_factory=ArgumentGraph)
    quality_scores: dict = Field(default_factory=dict)  # {round: {agent_id: QualityScores}}
    kappa: dict = Field(default_factory=dict)            # {round: CV value}
    sycophancy_flags: List[SycophancyFlag] = Field(default_factory=list)
    technique_convergence: Optional[float] = None       # τ⁽¹⁾
    divergence_flag: bool = False


# ─── Holistic Intelligent Decision Pack (Ω) ───────────────────────────────────
class ConfidenceProfile(BaseModel):
    C_est: float = Field(ge=0.0, le=1.0, description="Estimation stability")
    C_assump: float = Field(ge=0.0, le=1.0, description="Assumption solidity")
    C_complete: float = Field(ge=0.0, le=1.0, description="Scope completeness")
    C_consensus: float = Field(ge=0.0, le=1.0, description="Consensus genuineness")

    def rag_status(self, value: float) -> str:
        if value >= 0.70: return "GREEN"
        elif value >= 0.40: return "AMBER"
        return "RED"


class ScenarioEnvelope(BaseModel):
    optimistic_hours: float
    base_hours: float        # = consensus estimate Ê
    pessimistic_hours: float
    optimistic_conditions: str
    pessimistic_conditions: str


class AssumptionRegister(BaseModel):
    red_assumptions: List[str]    # high-contention, unresolved
    amber_assumptions: List[str]  # flagged by one agent, unverified
    green_assumptions: List[str]  # independently corroborated


class HolisticDecisionPack(BaseModel):
    # Ω₁
    consensus_estimate_hours: float
    ci_low_hours: float
    ci_high_hours: float
    # Ω₂
    confidence_profile: ConfidenceProfile
    # Ω₃
    scenario_envelope: ScenarioEnvelope
    # Ω₄
    assumption_register: AssumptionRegister
    # Ω₅
    recommendations: List[str]
    # Metadata
    technique_convergence: float
    sycophancy_rate: float
    rounds_to_converge: int
    divergence_flag: bool
    agent_weights: List[float]
    petsi_recommended_technique: str


# ─── PKD ─────────────────────────────────────────────────────────────────────
class ProjectKnowledgeDocument(BaseModel):
    pkd1_context: str = ""        # project charter, vision, scope narrative
    pkd2_scope: str = ""          # WBS, deliverables
    pkd3_resources: str = ""      # team, skills, constraints
    pkd4_visuals: str = ""        # Gantt, diagrams (absent in NASA93)
    pkd5_analogues: str = ""      # historical analogues (absent in NASA93)
    pkd6_known_unknowns: str = "" # PM-declared uncertainties

    def to_context_string(self) -> str:
        parts = []
        if self.pkd1_context:
            parts.append(f"PROJECT CONTEXT:\n{self.pkd1_context}")
        if self.pkd2_scope:
            parts.append(f"SCOPE:\n{self.pkd2_scope}")
        if self.pkd3_resources:
            parts.append(f"TEAM & RESOURCES:\n{self.pkd3_resources}")
        if self.pkd4_visuals:
            parts.append(f"VISUAL ARTEFACTS:\n{self.pkd4_visuals}")
        if self.pkd5_analogues:
            parts.append(f"HISTORICAL ANALOGUES:\n{self.pkd5_analogues}")
        if self.pkd6_known_unknowns:
            parts.append(f"KNOWN UNKNOWNS:\n{self.pkd6_known_unknowns}")
        return "\n\n".join(parts)
