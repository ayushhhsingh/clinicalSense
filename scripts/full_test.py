"""Quick smoke test – exercises the full ML pipeline on a handful of cases.

Run from the project root:

    python scripts/full_test.py
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.agents.medgemma_agents import ClinicalNote, ClinicalWorkflowOrchestrator


test_cases = [
    {
        "id": "TEST-001",
        "specialty": "Cardiology",
        "note": "Patient presents with acute chest pain, SOB. History of hypertension. BP 160/100. HR 105. EKG shows ST elevation in leads II, III. Troponin elevated. Patient transferred to cath lab.",
        "disease": "Coronary Artery Disease",
        "symptoms": ["chest pain", "shortness of breath"],
        "medications": ["Aspirin 81mg", "Atorvastatin 80mg"],
    },
    {
        "id": "TEST-002",
        "specialty": "Orthopedics",
        "note": "Patient with right knee pain after fall. Physical exam shows swelling and limited ROM. X-ray negative for fracture. Prescribed NSAIDs and physical therapy.",
        "disease": "Osteoarthritis",
        "symptoms": ["joint pain", "joint swelling"],
        "medications": ["Ibuprofen 400mg"],
    },
    {
        "id": "TEST-003",
        "specialty": "Pulmonology",
        "note": "Patient presents with fever, productive cough, shortness of breath. Chest X-ray shows consolidation. Started on empirical antibiotics.",
        "symptoms": ["fever", "cough", "shortness of breath"],
        "medications": [],
    },
    {
        "id": "TEST-004",
        "specialty": "Endocrinology",
        "note": "Patient with polyuria, polydipsia, and weight changes for 2 months. Labs show elevated HbA1c.",
        "disease": "Type 2 Diabetes",
        "symptoms": ["frequent urination", "excessive thirst", "fatigue"],
        "medications": ["Metformin 1000mg"],
    },
]


def main():
    orchestrator = ClinicalWorkflowOrchestrator()
    all_results = []
    for test in test_cases:
        print(f"\n=== {test['id']} ({test['specialty']}) ===")
        note = ClinicalNote(
            patient_id=test["id"],
            note_text=test.get("note", ""),
            disease_query=test.get("disease", ""),
            symptoms=test.get("symptoms", []),
            medications=test.get("medications", []),
        )
        result = orchestrator.process_note(note)
        result["specialty"] = test["specialty"]
        all_results.append(result)
        risks = result["workflow_stage_2_risks"]
        recs = result["workflow_stage_4_recommendations"]
        primary = result["workflow_stage_1_analysis"].get("primary_disease", {})
        print(
            f"  Disease: {primary.get('disease')} | "
            f"Risk: {risks.get('risk_score')} ({risks.get('risk_level')}) | "
            f"Urgency: {risks.get('urgency')}"
        )
        print(f"  Tools invoked: {result['tools_invoked']}")
        meds = recs.get("suggested_medications") or []
        print(f"  Suggested meds (first 3): {meds[:3]}")

    out_path = PROJECT_ROOT / "results" / "test_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved {len(all_results)} results to {out_path}")


if __name__ == "__main__":
    main()
