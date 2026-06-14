"""Flask API for the clinical-sense ML pipeline.

Endpoints
---------

``POST /analyze``
    Accepts a JSON body describing a patient (disease, symptoms, medications,
    free-text note) and returns the four-stage multi-agent analysis.

``GET /diseases``
    Lists all diseases available in the dataset.

``GET /diseases/search?q=...``
    Fuzzy search over the disease names.

``GET /agents``
    Returns metadata for the four agents.

``GET /health``
    Health check.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure the project root is on sys.path so ``backend.*`` imports resolve
# whether the file is invoked directly (`python backend/app.py`) or via a
# WSGI runner (`gunicorn backend.app:app`).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
from flask import Flask, jsonify, request  # noqa: E402
from flask_cors import CORS  # noqa: E402

from backend.agents.medgemma_agents import ClinicalNote, ClinicalWorkflowOrchestrator  # noqa: E402
from backend.utils.predictor import MLPredictor  # noqa: E402


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"


def _split_csv(value: str) -> List[str]:
    if not value:
        return []
    return [chunk.strip() for chunk in value.replace("\n", ",").split(",") if chunk.strip()]


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    predictor = MLPredictor(models_dir=MODELS_DIR, csv_path=DATA_DIR / "disease_dataset.csv")
    try:
        predictor.load()
        print(f"[startup] Loaded models from {MODELS_DIR}")
    except Exception as exc:  # noqa: BLE001
        print(f"[startup] Model load failed: {exc}")
        print("[startup] Run `python backend/utils/ml_models.py` to train models.")

    orchestrator = ClinicalWorkflowOrchestrator(predictor=predictor)

    # ------------------------------------------------------------------
    @app.route("/", methods=["GET"])
    def home():
        return jsonify({
            "status": "Ready",
            "version": "2.0",
            "ml_models_loaded": predictor._loaded,
            "agents": [
                "clinical_analyzer",
                "risk_detector",
                "drug_interaction",
                "recommendation",
            ],
        })

    # ------------------------------------------------------------------
    @app.route("/analyze", methods=["POST"])
    def analyze():
        try:
            data: Dict[str, Any] = request.get_json(silent=True) or {}

            note_text = (data.get("note_text") or "").strip()
            symptoms = data.get("symptoms") or []
            if isinstance(symptoms, str):
                symptoms = _split_csv(symptoms)
            symptoms = [str(s).strip() for s in symptoms if str(s).strip()]

            medications = data.get("medications") or []
            if isinstance(medications, str):
                medications = _split_csv(medications)
            medications = [str(m).strip() for m in medications if str(m).strip()]

            disease_query = (data.get("disease") or data.get("disease_query") or "").strip()

            if not (note_text or symptoms or disease_query):
                return jsonify({
                    "success": False,
                    "error": "Provide at least one of: 'disease', 'symptoms', or 'note_text'.",
                }), 400

            note = ClinicalNote(
                patient_id=str(data.get("patient_id", "DEMO-001")),
                note_text=note_text,
                symptoms=symptoms,
                disease_query=disease_query,
                medications=medications,
            )
            result = orchestrator.process_note(note)
            return jsonify({"success": True, "data": result})

        except FileNotFoundError as exc:
            return jsonify({
                "success": False,
                "error": f"Dataset or model artefact missing: {exc}. Run the training script first.",
            }), 500
        except Exception as exc:  # noqa: BLE001
            return jsonify({"success": False, "error": str(exc)}), 500

    # ------------------------------------------------------------------
    @app.route("/diseases", methods=["GET"])
    def list_diseases():
        if predictor.dataset is None:
            return jsonify({"success": False, "error": "Dataset not loaded"}), 500
        records = []
        for _, row in predictor.dataset.iterrows():
            records.append({
                "disease": row["disease"],
                "specialty": row.get("specialty", ""),
                "severity": row.get("severity", ""),
                "risk_score": int(row.get("risk_score", 0) or 0),
                "symptom_count": int(row.get("symptom_count", 0) or 0),
                "medication_count": int(row.get("med_count", 0) or 0),
            })
        return jsonify({"success": True, "count": len(records), "diseases": records})

    # ------------------------------------------------------------------
    @app.route("/diseases/search", methods=["GET"])
    def search_diseases():
        query = (request.args.get("q") or "").strip()
        if not query:
            return jsonify({"success": False, "error": "Missing query parameter 'q'"}), 400
        results = predictor.search_disease(query, k=8)
        return jsonify({"success": True, "query": query, "results": results})

    # ------------------------------------------------------------------
    @app.route("/agents", methods=["GET"])
    def get_agents():
        return jsonify({
            "success": True,
            "agents": [
                {
                    "name": "Clinical Analyzer",
                    "type": "clinical_analyzer",
                    "tools": ["extract_symptoms", "predict_disease", "cluster_disease"],
                    "models": ["RandomForestClassifier", "KMeans"],
                },
                {
                    "name": "Risk Detector",
                    "type": "risk_detector",
                    "tools": ["score_risk", "assess_severity", "flag_critical"],
                    "models": ["GradientBoostingRegressor"],
                },
                {
                    "name": "Drug Interaction",
                    "type": "drug_interaction",
                    "tools": ["check_interactions", "check_contraindications", "verify_dosage"],
                    "models": ["Static drug-interaction lookup"],
                },
                {
                    "name": "Recommendation",
                    "type": "recommendation",
                    "tools": ["suggest_medications", "suggest_tests", "predict_outcomes"],
                    "models": ["Dataset-driven recommendation"],
                },
            ],
        })

    # ------------------------------------------------------------------
    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "healthy",
            "models_loaded": predictor._loaded,
            "vocabulary_size": len(predictor.vocabulary) if predictor.vocabulary else 0,
        }), 200

    return app


app = create_app()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(debug=False, port=port, host="0.0.0.0")
