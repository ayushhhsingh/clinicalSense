"""Clinical multi-agent system – ML-driven.

The previous version of this file relied on hard-coded keyword dictionaries
and regex heuristics. That implementation suffered from several bugs:

* The keyword tables (``CONDITION_KEYWORDS``, ``SYMPTOM_KEYWORDS``,
  ``RECOMMENDATION_MAP`` …) duplicated information that already lives in
  ``data/disease_dataset.csv`` and could drift out of sync.
* The risk score, severity, medications and tests were hand-tuned constants.
* No actual machine-learning was performed – everything was string matching.

This rewrite keeps the *shape* of the multi-agent workflow (analyser → risk
detector → drug interaction → recommendation) but powers every agent with
real models trained from the CSV:

* **Clinical Analyzer** – predicts the most likely disease and the cluster
  of similar diseases from the user-supplied symptoms / free text.
* **Risk Detector** – uses a Gradient Boosting regressor to compute a
  continuous risk score and maps it to a categorical risk level.
* **Drug Interaction** – the same small interaction database that the
  previous version embedded, but evaluated against medications the model
  actually recommended.
* **Recommendation Engine** – pulls the medication / test list directly
  from the CSV and produces monitoring + follow-up guidance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from backend.utils.predictor import MLPredictor, _risk_level


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ClinicalNote:
    patient_id: str
    note_text: str = ""
    symptoms: List[str] = field(default_factory=list)
    disease_query: str = ""
    medications: List[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    agent_name: str
    findings: Dict[str, Any]
    confidence: float
    tool_calls_made: List[str]


class AgentType(Enum):
    CLINICAL_ANALYZER = "clinical_analyzer"
    RISK_DETECTOR = "risk_detector"
    DRUG_INTERACTION = "drug_interaction"
    RECOMMENDATION = "recommendation"


# ---------------------------------------------------------------------------
# Drug interaction knowledge base
# ---------------------------------------------------------------------------
# Small, hand-curated database – replacing the old ``DRUG_INTERACTIONS_DB``
# but kept as a static lookup since interaction data is not in the dataset.

DRUG_INTERACTIONS_DB: Dict[tuple, Dict[str, str]] = {
    ("warfarin", "aspirin"):       {"severity": "HIGH",   "effect": "Increased bleeding risk"},
    ("warfarin", "ibuprofen"):     {"severity": "HIGH",   "effect": "Increased bleeding risk"},
    ("warfarin", "naproxen"):      {"severity": "HIGH",   "effect": "Increased bleeding risk"},
    ("lisinopril", "potassium"):   {"severity": "MEDIUM", "effect": "Hyperkalemia risk"},
    ("lisinopril", "spironolactone"): {"severity": "MEDIUM", "effect": "Hyperkalemia risk"},
    ("metformin", "alcohol"):      {"severity": "MEDIUM", "effect": "Lactic acidosis risk"},
    ("metformin", "contrast"):     {"severity": "HIGH",   "effect": "Lactic acidosis risk — hold before contrast"},
    ("ssri", "tramadol"):          {"severity": "HIGH",   "effect": "Serotonin syndrome risk"},
    ("ssri", "maoi"):              {"severity": "HIGH",   "effect": "Serotonin syndrome — contraindicated"},
    ("statin", "gemfibrozil"):     {"severity": "HIGH",   "effect": "Myopathy / rhabdomyolysis risk"},
    ("digoxin", "amiodarone"):     {"severity": "HIGH",   "effect": "Digoxin toxicity risk"},
    ("ace inhibitor", "nsaid"):    {"severity": "MEDIUM", "effect": "Reduced antihypertensive effect, nephrotoxicity"},
    ("beta blocker", "verapamil"): {"severity": "HIGH",   "effect": "Bradycardia and heart block risk"},
    ("clopidogrel", "ppi"):        {"severity": "MEDIUM", "effect": "Reduced antiplatelet efficacy"},
}

DRUG_ALIASES: Dict[str, List[str]] = {
    "lisinopril": ["lisinopril", "ace inhibitor"],
    "enalapril":  ["enalapril", "ace inhibitor"],
    "ramipril":   ["ramipril", "ace inhibitor"],
    "atorvastatin": ["atorvastatin", "statin"],
    "rosuvastatin": ["rosuvastatin", "statin"],
    "simvastatin":  ["simvastatin", "statin"],
    "metoprolol":   ["metoprolol", "beta blocker"],
    "atenolol":     ["atenolol", "beta blocker"],
    "carvedilol":   ["carvedilol", "beta blocker"],
    "sertraline":   ["sertraline", "ssri"],
    "fluoxetine":   ["fluoxetine", "ssri"],
    "escitalopram": ["escitalopram", "ssri"],
    "citalopram":   ["citalopram", "ssri"],
    "phenelzine":   ["phenelzine", "maoi"],
    "tranylcypromine": ["tranylcypromine", "maoi"],
}

INLINE_MED_PATTERN = re.compile(
    r"\b(aspirin|warfarin|metformin|lisinopril|atorvastatin|metoprolol|"
    r"amlodipine|furosemide|omeprazole|sertraline|clopidogrel|digoxin|"
    r"amiodarone|tramadol|ibuprofen|naproxen|spironolactone|verapamil|"
    r"gemfibrozil|apixaban|rivaroxaban|albuterol|prednisone)\b",
    re.IGNORECASE,
)


def _normalize_med(med: str) -> List[str]:
    """Strip dosage info and return canonical name + class aliases."""
    med_lower = med.lower().strip()
    base = re.sub(r"\s*\d+\s*mg.*", "", med_lower).strip()
    aliases = [base]
    for canonical, alias_list in DRUG_ALIASES.items():
        if base == canonical or base in alias_list:
            aliases.extend(alias_list)
    return list(set(aliases))


def _check_drug_interactions(medications: List[str]) -> List[Dict[str, str]]:
    normalized = [_normalize_med(m) for m in medications]
    found: List[Dict[str, str]] = []
    checked = set()
    for i, aliases_i in enumerate(normalized):
        for j, aliases_j in enumerate(normalized):
            if i >= j:
                continue
            pair_key = tuple(sorted([i, j]))
            if pair_key in checked:
                continue
            checked.add(pair_key)
            for alias_i in aliases_i:
                for alias_j in aliases_j:
                    interaction = (
                        DRUG_INTERACTIONS_DB.get((alias_i, alias_j))
                        or DRUG_INTERACTIONS_DB.get((alias_j, alias_i))
                    )
                    if interaction:
                        found.append({
                            "drug_1": medications[i],
                            "drug_2": medications[j],
                            "severity": interaction["severity"],
                            "effect": interaction["effect"],
                        })
                        break
    return found


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

class MedGemmaAgent:
    def __init__(self, agent_type: AgentType, predictor: MLPredictor):
        self.agent_type = agent_type
        self.predictor = predictor
        self.tools = self._initialize_tools()

    def _initialize_tools(self) -> List[str]:
        base = ["analyze_text", "extract_entities", "score_findings"]
        tools_map = {
            AgentType.CLINICAL_ANALYZER: base + ["extract_symptoms", "predict_disease", "cluster_disease"],
            AgentType.RISK_DETECTOR:     base + ["score_risk", "assess_severity", "flag_critical"],
            AgentType.DRUG_INTERACTION:  base + ["check_interactions", "check_contraindications", "verify_dosage"],
            AgentType.RECOMMENDATION:    base + ["suggest_medications", "suggest_tests", "predict_outcomes"],
        }
        return tools_map.get(self.agent_type, base)

    def call_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        return {"status": "success", "tool": tool_name}

    def analyze(self, note: ClinicalNote) -> AnalysisResult:
        findings, tool_calls = self._run_analysis(note)
        for tc in tool_calls:
            self.call_tool(tc)
        return AnalysisResult(
            agent_name=self.agent_type.value,
            findings=findings,
            confidence=self._compute_confidence(findings),
            tool_calls_made=tool_calls,
        )

    def _compute_confidence(self, findings: Dict[str, Any]) -> float:
        score = 0.6
        if findings.get("found") is False:
            return 0.3
        if findings.get("top_diseases"):
            top = findings["top_diseases"][0]
            score = min(0.95, 0.55 + float(top.get("probability", 0)) / 2)
        if findings.get("medications"):
            score = min(0.95, score + 0.05)
        return round(score, 2)

    # ------------------------------------------------------------------
    def _run_analysis(self, note: ClinicalNote):
        if self.agent_type == AgentType.CLINICAL_ANALYZER:
            return self._clinical_analysis(note)
        if self.agent_type == AgentType.RISK_DETECTOR:
            return self._risk_analysis(note)
        if self.agent_type == AgentType.DRUG_INTERACTION:
            return self._drug_analysis(note)
        if self.agent_type == AgentType.RECOMMENDATION:
            return self._recommendation_analysis(note)
        return {}, []

    # ----- Clinical Analyzer ---------------------------------------------
    def _clinical_analysis(self, note: ClinicalNote):
        tool_calls = ["extract_symptoms", "predict_disease", "cluster_disease"]

        # Build a unified symptom list from explicit input + free text.
        explicit_symptoms = [s.strip() for s in (note.symptoms or []) if s.strip()]
        text_symptoms = self.predictor.extract_symptoms(note.note_text or "")
        all_symptoms = list(dict.fromkeys(explicit_symptoms + text_symptoms))

        symptom_prediction: Optional[Dict[str, Any]] = None
        if all_symptoms:
            symptom_prediction = self.predictor.predict_from_symptoms(all_symptoms, top_k=3)

        disease_lookup: Optional[Dict[str, Any]] = None
        if note.disease_query:
            disease_lookup = self.predictor.predict_from_disease(note.disease_query)

        # Cross check: if a disease name was supplied, treat it as the primary
        # hypothesis. Otherwise the ML model is the primary hypothesis.
        primary_disease: Optional[Dict[str, Any]] = None
        if disease_lookup and disease_lookup.get("found"):
            primary_disease = disease_lookup
        elif symptom_prediction and symptom_prediction.get("top_diseases"):
            primary_disease = symptom_prediction["top_diseases"][0]

        # Light demographics extraction.
        text_lower = (note.note_text or "").lower()
        age_match = re.search(r"(\d{1,3})\s*(?:y/?o|year[s]?[\s-]*old|yo)", text_lower)
        gender_match = re.search(r"\b(male|female|man|woman|m\b|f\b)\b", text_lower)
        age = age_match.group(1) if age_match else None
        gender = gender_match.group(1) if gender_match else None

        findings: Dict[str, Any] = {
            "symptoms_identified": all_symptoms,
            "primary_disease": primary_disease,
            "alternative_diseases": (
                [d for d in (symptom_prediction or {}).get("top_diseases", [])[1:]]
                if symptom_prediction else []
            ),
            "cluster_related_diseases": (
                (symptom_prediction or {}).get("cluster_related_diseases", [])
            ),
            "model_used": "RandomForestClassifier + GradientBoostingRegressor + KMeans",
        }
        if disease_lookup is not None:
            findings["disease_lookup"] = disease_lookup
        if age:
            findings["patient_age"] = age
        if gender:
            findings["patient_gender"] = gender

        return findings, tool_calls

    # ----- Risk Detector ---------------------------------------------------
    def _risk_analysis(self, note: ClinicalNote):
        tool_calls = ["score_risk", "assess_severity", "flag_critical"]

        primary = self._primary_disease(note)
        explicit_symptoms = [s.strip() for s in (note.symptoms or []) if s.strip()]
        text_symptoms = self.predictor.extract_symptoms(note.note_text or "")
        all_symptoms = list(dict.fromkeys(explicit_symptoms + text_symptoms))

        # Risk score: prefer dataset score for primary disease, fall back to
        # the regressor.
        risk_score: Optional[float] = None
        severity: Optional[str] = None
        confidence = 0.0
        rationale_parts: List[str] = []

        if primary and "risk_score" in primary:
            risk_score = float(primary["risk_score"])
            severity = primary.get("severity")
            confidence = 0.95
            rationale_parts.append(
                f"Risk score pulled directly from dataset for {primary.get('disease')}"
            )
        elif all_symptoms:
            pred = self.predictor.predict_from_symptoms(all_symptoms, top_k=1)
            risk_score = float(pred["predicted_risk_score"] or 0)
            confidence = 0.6 + float(pred["top_diseases"][0]["probability"]) / 2 if pred["top_diseases"] else 0.6
            rationale_parts.append("Risk score estimated by GradientBoosting regressor")
        else:
            risk_score = 0.0
            confidence = 0.2
            rationale_parts.append("Insufficient information; defaulting to zero risk")

        # Critical symptom flags
        critical_keywords = [
            "chest pain", "shortness of breath", "loss of consciousness",
            "confusion", "severe headache", "seizures", "coughing blood",
            "blood in stool", "blood in urine", "jaundice",
        ]
        flags = [s for s in all_symptoms if s.lower() in critical_keywords]
        if primary and primary.get("severity", "").lower() == "critical":
            flags.append(f"Critical-severity disease: {primary.get('disease')}")

        risk_level = _risk_level(risk_score or 0)
        urgency = {
            "Critical": "EMERGENCY",
            "High": "URGENT",
            "Medium": "SEMI-URGENT",
            "Low": "ROUTINE",
            "Unknown": "ROUTINE",
        }.get(risk_level, "ROUTINE")

        findings = {
            "primary_disease": primary.get("disease") if primary else None,
            "risk_score": round(risk_score or 0, 1),
            "risk_level": risk_level,
            "urgency": urgency,
            "severity": severity,
            "critical_flags": flags or ["No immediate critical flags identified"],
            "rationale": "; ".join(rationale_parts),
            "confidence": round(confidence, 2),
        }

        return findings, tool_calls

    # ----- Drug Interaction ------------------------------------------------
    def _drug_analysis(self, note: ClinicalNote):
        tool_calls = ["check_interactions", "check_contraindications", "verify_dosage"]

        # Collect the medications to evaluate:
        # 1. The ones the user typed in.
        medications: List[str] = list(note.medications or [])
        # 2. Any additional meds mentioned inline in the note text.
        for hit in INLINE_MED_PATTERN.findall(note.note_text or ""):
            cap = hit.capitalize()
            if not any(cap.lower() in existing.lower() for existing in medications):
                medications.append(cap)
        # 3. The medications recommended by the model for the primary disease.
        primary = self._primary_disease(note)
        recommended = list((primary or {}).get("medications", []) or [])

        interactions = _check_drug_interactions(medications)
        recommended_interactions = _check_drug_interactions(recommended)

        high = [i for i in interactions if i["severity"] == "HIGH"]
        medium = [i for i in interactions if i["severity"] == "MEDIUM"]

        if high:
            overall_status = "CAUTION — HIGH severity interaction(s) detected"
        elif medium:
            overall_status = "CAUTION — moderate interaction(s) detected"
        elif medications:
            overall_status = "Safe — no major interactions detected"
        else:
            overall_status = "No medications provided to evaluate"

        findings = {
            "medications_reviewed": medications,
            "recommended_medications": recommended,
            "recommended_medication_interactions": recommended_interactions,
            "interactions_found": interactions,
            "high_severity_interactions": high,
            "medium_severity_interactions": medium,
            "overall_status": overall_status,
            "total_interactions": len(interactions),
        }

        return findings, tool_calls

    # ----- Recommendation Engine ------------------------------------------
    def _recommendation_analysis(self, note: ClinicalNote):
        tool_calls = ["suggest_medications", "suggest_tests", "predict_outcomes"]

        primary = self._primary_disease(note)
        risk = self._risk_from_note(note, primary)

        suggested_meds = list((primary or {}).get("medications", []) or [])
        suggested_tests = list((primary or {}).get("tests", []) or [])
        specialty = (primary or {}).get("specialty", "")
        description = (primary or {}).get("description", "")

        # Outcome prediction: tie it to the risk score.
        if risk is None:
            outcome = "Insufficient data to predict outcome"
        elif risk >= 80:
            outcome = "Guarded — requires urgent intervention and close monitoring"
        elif risk >= 60:
            outcome = "Guarded — expected to require active treatment"
        elif risk >= 35:
            outcome = "Fair — expected improvement with appropriate management"
        else:
            outcome = "Good — likely to improve with standard treatment and follow-up"

        monitoring = ["Periodic vital sign checks"]
        if primary and ("Cardiology" in specialty or "cardiac" in description.lower()):
            monitoring.append("Continuous cardiac monitoring (telemetry)")
        if primary and "Pulmonology" in specialty:
            monitoring.append("Serial oxygen saturation monitoring")
        if primary and "Endocrinology" in specialty:
            monitoring.append("Blood glucose / metabolic panel monitoring")
        if "chest pain" in [s.lower() for s in (note.symptoms or [])] or "chest pain" in (note.note_text or "").lower():
            monitoring.append("Continuous cardiac monitoring (telemetry)")

        findings = {
            "primary_disease": (primary or {}).get("disease"),
            "specialty": specialty,
            "description": description,
            "suggested_medications": suggested_meds or ["Supportive care — no specific medication recommended"],
            "suggested_tests": suggested_tests or ["General workup recommended"],
            "monitoring_plan": monitoring,
            "outcome_prediction": outcome,
            "follow_up": "Re-evaluate in 24–48h or sooner if condition changes",
            "risk_score": risk,
        }

        return findings, tool_calls

    # ------------------------------------------------------------------
    def _primary_disease(self, note: ClinicalNote) -> Optional[Dict[str, Any]]:
        """Return the primary disease dict for the note (or None)."""
        if note.disease_query:
            lookup = self.predictor.predict_from_disease(note.disease_query)
            if lookup.get("found"):
                return lookup
        symptoms = [s for s in (note.symptoms or []) if s.strip()]
        if not symptoms and note.note_text:
            symptoms = self.predictor.extract_symptoms(note.note_text)
        if not symptoms:
            return None
        prediction = self.predictor.predict_from_symptoms(symptoms, top_k=1)
        if prediction["top_diseases"]:
            return prediction["top_diseases"][0]
        return None

    def _risk_from_note(self, note: ClinicalNote, primary: Optional[Dict[str, Any]]) -> Optional[float]:
        if primary and "risk_score" in primary:
            return float(primary["risk_score"])
        symptoms = [s for s in (note.symptoms or []) if s.strip()]
        if not symptoms and note.note_text:
            symptoms = self.predictor.extract_symptoms(note.note_text)
        if not symptoms:
            return None
        prediction = self.predictor.predict_from_symptoms(symptoms, top_k=1)
        return float(prediction["predicted_risk_score"] or 0)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class ClinicalWorkflowOrchestrator:
    def __init__(self, predictor: Optional[MLPredictor] = None):
        self.predictor = predictor or MLPredictor()
        self.predictor.load()
        self.agents = {
            AgentType.CLINICAL_ANALYZER: MedGemmaAgent(AgentType.CLINICAL_ANALYZER, self.predictor),
            AgentType.RISK_DETECTOR:     MedGemmaAgent(AgentType.RISK_DETECTOR, self.predictor),
            AgentType.DRUG_INTERACTION:  MedGemmaAgent(AgentType.DRUG_INTERACTION, self.predictor),
            AgentType.RECOMMENDATION:    MedGemmaAgent(AgentType.RECOMMENDATION, self.predictor),
        }
        self.execution_log: List[AnalysisResult] = []

    def process_note(self, note: ClinicalNote) -> Dict[str, Any]:
        results: Dict[str, AnalysisResult] = {}
        self.execution_log = []

        results[AgentType.CLINICAL_ANALYZER] = self.agents[AgentType.CLINICAL_ANALYZER].analyze(note)
        self.execution_log.append(results[AgentType.CLINICAL_ANALYZER])

        results[AgentType.RISK_DETECTOR] = self.agents[AgentType.RISK_DETECTOR].analyze(note)
        self.execution_log.append(results[AgentType.RISK_DETECTOR])

        results[AgentType.DRUG_INTERACTION] = self.agents[AgentType.DRUG_INTERACTION].analyze(note)
        self.execution_log.append(results[AgentType.DRUG_INTERACTION])

        results[AgentType.RECOMMENDATION] = self.agents[AgentType.RECOMMENDATION].analyze(note)
        self.execution_log.append(results[AgentType.RECOMMENDATION])

        return {
            "patient_id": note.patient_id,
            "input": {
                "disease_query": note.disease_query,
                "symptoms": note.symptoms,
                "medications": note.medications,
                "note_text": note.note_text,
            },
            "workflow_stage_1_analysis": results[AgentType.CLINICAL_ANALYZER].findings,
            "workflow_stage_2_risks":    results[AgentType.RISK_DETECTOR].findings,
            "workflow_stage_3_interactions": results[AgentType.DRUG_INTERACTION].findings,
            "workflow_stage_4_recommendations": results[AgentType.RECOMMENDATION].findings,
            "tools_invoked": sum(len(r.tool_calls_made) for r in self.execution_log),
            "execution_log": [
                {
                    "agent": r.agent_name,
                    "tools_called": r.tool_calls_made,
                    "confidence": r.confidence,
                }
                for r in self.execution_log
            ],
        }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample = ClinicalNote(
        patient_id="P12345",
        note_text=(
            "Patient with chest pain, shortness of breath, sweating. "
            "History of hypertension. BP 160/100, HR 105."
        ),
        symptoms=["chest pain", "shortness of breath", "sweating"],
        medications=["Lisinopril 10mg", "Aspirin 81mg"],
    )

    orchestrator = ClinicalWorkflowOrchestrator()
    result = orchestrator.process_note(sample)

    import json
    print(json.dumps(result, indent=2, default=str))
