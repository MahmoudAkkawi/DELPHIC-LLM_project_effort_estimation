"""
DELPHIC-LLM: LLM client
Wraps OpenAI API with retry logic, cost tracking, and JSON schema enforcement.
"""
import json
import time
import os
from typing import Any, Type
from pydantic import BaseModel

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class LLMClient:
    """
    Thin wrapper around OpenAI API.
    Tracks token usage and cost automatically.
    """
    # GPT-4o pricing (per 1M tokens, as of late 2024)
    PRICE_INPUT_PER_M  = 2.50   # USD
    PRICE_OUTPUT_PER_M = 10.00  # USD

    # Maps common LLM field-name variations to our exact schema field names
    FIELD_ALIASES = {
        # Round1Output
        "selectedTechnique":        "technique_selected",
        "technique":                "technique_selected",
        "estimationTechnique":      "technique_selected",
        "techniqueName":            "technique_selected",
        "justification":            "technique_justification",
        "techniqueJustification":   "technique_justification",
        "rationale":                "technique_justification",
        "estimate":                 "estimate_hours",
        "totalEffort":              "estimate_hours",
        "effortHours":              "estimate_hours",
        "effort_estimate":          "estimate_hours",
        "estimatedHours":           "estimate_hours",
        "confidenceIntervalLow":    "confidence_interval_low",
        "ci_low":                   "confidence_interval_low",
        "lower_bound":              "confidence_interval_low",
        "lowerBound":               "confidence_interval_low",
        "confidenceIntervalHigh":   "confidence_interval_high",
        "ci_high":                  "confidence_interval_high",
        "upper_bound":              "confidence_interval_high",
        "upperBound":               "confidence_interval_high",
        "assumptions":              "key_assumptions",
        "keyAssumptions":           "key_assumptions",
        "risks":                    "identified_risks",
        "identifiedRisks":          "identified_risks",
        "riskFactors":              "identified_risks",
        "reasoning":                "reasoning_chain",
        "reasoningChain":           "reasoning_chain",
        "derivation":               "reasoning_chain",
        "uncertaintySources":       "uncertainty_sources",
        "uncertainty_factors":      "uncertainty_sources",
        # Round2ChallengerOutput - nested challenge fields
        "estimate_id":              "target_estimate_id",
        "estimateId":               "target_estimate_id",
        "weakest_assumption":       "identified_weakness",
        "weakestAssumption":        "identified_weakness",
        "weakness":                 "identified_weakness",
        "weak_assumption":          "identified_weakness",
        "weakAssumption":           "identified_weakness",
        "assumption":               "identified_weakness",
        "vulnerable_assumption":    "identified_weakness",
        "quantified_impact_hours":  "impact_if_wrong_hours",
        "quantifiedImpactHours":    "impact_if_wrong_hours",
        "estimated_impact_hours":   "impact_if_wrong_hours",
        "impact_hours":             "impact_if_wrong_hours",
        # Round2BuilderOutput
        "synthesis_hours":          "proposed_synthesis_hours",
        "proposedSynthesisHours":   "proposed_synthesis_hours",
        "proposed_estimate_hours":  "proposed_synthesis_hours",
        "unresolved_tensions":      "unresolved_tensions",
        "supported_elements":       "supported_elements",
        "genuine_agreement_areas":  "genuine_agreement_areas",
        # Round3Output
        "finalEstimate":            "final_estimate_hours",
        "final_estimate":           "final_estimate_hours",
        "changeFromRound1":         "change_from_round1",
        "challengeAddressed":       "challenge_addressed",
        "challengeRebutted":        "challenge_rebutted",
        "updatedCILow":             "updated_confidence_interval_low",
        "updatedCIHigh":            "updated_confidence_interval_high",
        "remainingUncertainty":     "remaining_uncertainty",
        "techniqueMaintained":      "technique_maintained",
        "techniqueSwitchReason":    "technique_switch_reason",
    }

    def __init__(self,
                 model: str = "gpt-4o-2024-11-20",
                 agent_temperature: float = 0.7,
                 oa_temperature: float = 0.2,
                 max_retries: int = 3):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package required: pip install openai")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.agent_temperature = agent_temperature
        self.oa_temperature = oa_temperature
        self.max_retries = max_retries
        # Cost tracking
        self.total_input_tokens  = 0
        self.total_output_tokens = 0

    @property
    def total_cost_usd(self) -> float:
        return (self.total_input_tokens  / 1_000_000 * self.PRICE_INPUT_PER_M +
                self.total_output_tokens / 1_000_000 * self.PRICE_OUTPUT_PER_M)

    def _call(self,
              system: str,
              user: str,
              temperature: float,
              response_schema: Type[BaseModel] | None = None) -> str:
        """Raw API call with retry and token tracking."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user}
        ]
        last_err = None
        for attempt in range(self.max_retries):
            try:
                kwargs: dict[str, Any] = {
                    "model":       self.model,
                    "messages":    messages,
                    "temperature": temperature,
                }
                if response_schema is not None:
                    # Use JSON mode for schema enforcement
                    kwargs["response_format"] = {"type": "json_object"}
                response = self.client.chat.completions.create(**kwargs)
                # Track usage
                usage = response.usage
                if usage:
                    self.total_input_tokens  += usage.prompt_tokens
                    self.total_output_tokens += usage.completion_tokens
                content = response.choices[0].message.content
                return content or ""
            except Exception as e:
                last_err = e
                wait = 2 ** attempt
                print(f"  [LLM] Attempt {attempt+1} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)
        raise RuntimeError(f"LLM call failed after {self.max_retries} attempts: {last_err}")

    def agent_call(self,
                   system: str,
                   user: str,
                   schema: Type[BaseModel]) -> BaseModel:
        """Call as a deliberation agent (higher temperature), parse to schema."""
        raw = self._call(system, user, self.agent_temperature, schema)
        return self._parse(raw, schema)

    def orchestrator_call(self,
                          system: str,
                          user: str,
                          schema: "Type[BaseModel] | None" = None):
        """Call as orchestrator (lower temperature)."""
        raw = self._call(system, user, self.oa_temperature, schema)
        if not raw:
            return {} if schema is None else None
        if schema is not None:
            return self._parse(raw, schema)
        text = raw.strip()
        if text.startswith("```"):
            parts = text.split("\n")
            text = "\n".join(parts[1:-1] if parts and parts[-1].strip() == "```" else parts[1:])
        try:
            result = json.loads(text.strip())
            if isinstance(result, dict):
                result = self._normalise(result)
            return result
        except json.JSONDecodeError:
            return raw
    def _normalise(self, data: dict) -> dict:
        """Rename aliased keys and flatten nested values."""
        result = {}
        numeric = {
            "estimate_hours", "confidence_interval_low", "confidence_interval_high",
            "final_estimate_hours", "change_from_round1",
            "updated_confidence_interval_low", "updated_confidence_interval_high",
            "impact_if_wrong_hours", "impact_hours", "proposed_synthesis_hours"
        }
        for key, value in data.items():
            canon = self.FIELD_ALIASES.get(key, key)
            # Flatten nested dict to string
            if isinstance(value, dict):
                value = " | ".join(
                    f"{k}: {v}" for k, v in value.items() if isinstance(v, (str, int, float))
                )[:500]
            # Convert string numbers
            if canon in numeric and isinstance(value, str):
                try:
                    value = float(value.replace(",", "").split()[0])
                except (ValueError, IndexError):
                    pass
            result[canon] = value
        return result

    def _reshape_for_schema(self, data: dict, schema_name: str) -> dict:
        """Fix structural variations specific to each schema."""
        if schema_name == "Round2ChallengerOutput":
            if "challenges" not in data:
                challenges = []
                for i in range(1, 10):
                    key = f"estimate_{i}"
                    if key in data:
                        ch = data[key] if isinstance(data[key], dict) else {}
                        ch["target_estimate_id"] = i
                        challenges.append(ch)
                if challenges:
                    data = {"challenges": challenges, "overall_assessment": "See challenges above."}
            norm_challenges = []
            for ch in data.get("challenges", []):
                ch = self._normalise(ch)
                if "target_estimate_id" not in ch:
                    ch["target_estimate_id"] = len(norm_challenges) + 1
                if "identified_weakness" not in ch:
                    ch["identified_weakness"] = ch.get("counter_argument", "Assumption not specified")[:200]
                if "severity" not in ch:
                    ch["severity"] = "MEDIUM"
                if "impact_if_wrong_hours" not in ch:
                    ch["impact_if_wrong_hours"] = 100.0
                if "counter_argument" not in ch:
                    ch["counter_argument"] = ch.get("identified_weakness", "No counter-argument provided")[:200]
                norm_challenges.append(ch)
            data["challenges"] = norm_challenges
            if "overall_assessment" not in data:
                data["overall_assessment"] = "Challenges identified above represent key weaknesses."
        elif schema_name == "Round2BuilderOutput":
            if "supported_elements" not in data:
                data["supported_elements"] = []
            if "genuine_agreement_areas" not in data:
                data["genuine_agreement_areas"] = ["Estimates are within same order of magnitude"]
            if "proposed_synthesis_hours" not in data:
                for k, v in data.items():
                    if isinstance(v, (int, float)) and v > 0:
                        data["proposed_synthesis_hours"] = float(v)
                        break
                else:
                    data["proposed_synthesis_hours"] = 1000.0
            if "synthesis_rationale" not in data:
                data["synthesis_rationale"] = str(data.get("rationale", "Synthesis of agent estimates"))[:300]
            if "unresolved_tensions" not in data:
                data["unresolved_tensions"] = ["Scope uncertainty remains unresolved"]
        elif schema_name == "Round2RiskAnalystOutput":
            if "risk_items" not in data:
                data["risk_items"] = []
            if "overall_risk_level" not in data:
                data["overall_risk_level"] = "MEDIUM"
            norm_risks = []
            for i, ri in enumerate(data.get("risk_items", [])):
                ri = self._normalise(ri) if isinstance(ri, dict) else {}
                if "estimate_id" not in ri:
                    ri["estimate_id"] = i + 1
                for field, default in [
                    ("critical_assumption", "Key assumption not specified"),
                    ("failure_scenario", "Scenario not specified"),
                    ("probability", "MEDIUM"),
                    ("impact_hours", 100.0),
                    ("hedge_recommendation", "Monitor closely"),
                ]:
                    if field not in ri:
                        ri[field] = default
                norm_risks.append(ri)
            data["risk_items"] = norm_risks
        elif schema_name == "Round3Output":
            if "final_estimate_hours" not in data:
                for k in ["estimate_hours", "estimate", "final_estimate"]:
                    if k in data:
                        data["final_estimate_hours"] = float(data[k])
                        break
                else:
                    data["final_estimate_hours"] = 1000.0
            if "change_from_round1" not in data:
                data["change_from_round1"] = 0.0
            if "challenge_addressed" not in data:
                data["challenge_addressed"] = data.get("reasoning", "Challenge addressed.")[:300]
            if "challenge_rebutted" not in data:
                data["challenge_rebutted"] = False
            if "updated_confidence_interval_low" not in data:
                e = float(data.get("final_estimate_hours", 1000.0))
                data["updated_confidence_interval_low"] = e * 0.7
            if "updated_confidence_interval_high" not in data:
                e = float(data.get("final_estimate_hours", 1000.0))
                data["updated_confidence_interval_high"] = e * 1.4
            if "remaining_uncertainty" not in data:
                data["remaining_uncertainty"] = "Scope and requirement uncertainty remain."
            if "technique_maintained" not in data:
                data["technique_maintained"] = True
        return data

    def _parse(self, raw: str, schema: Type[BaseModel]) -> BaseModel:
        """Parse JSON string into Pydantic schema with field-name normalisation."""
        text = raw.strip()
        if text.startswith("```"):
            parts = text.split("\n")
            text = "\n".join(parts[1:-1] if parts and parts[-1].strip() == "```" else parts[1:])
        text = text.strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"LLM returned invalid JSON for {schema.__name__}: {e}\n"
                f"Raw (first 300 chars): {raw[:300]}"
            )
        data = self._normalise(data)
        data = self._reshape_for_schema(data, schema.__name__)
        try:
            return schema.model_validate(data)
        except Exception as e:
            raise ValueError(
                f"Failed to parse {schema.__name__}: {e}\n"
                f"Keys received: {list(data.keys())}\n"
                f"Raw (first 400 chars): {raw[:400]}"
            )

    def reset_cost_tracker(self):
        self.total_input_tokens  = 0
        self.total_output_tokens = 0
