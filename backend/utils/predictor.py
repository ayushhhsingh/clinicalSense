"""Inference wrapper around the trained artefacts.

The ``MLPredictor`` class encapsulates loading the persisted models, taking a
list of symptoms or a free-text note, and returning a structured prediction
containing the most likely disease, the predicted risk score, the cluster
membership and the dataset-derived medication + test recommendations.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from backend.utils.data_processor import (
    DEFAULT_DATASET_PATH,
    MODELS_DIR,
    build_symptom_vocabulary,
    load_dataset,
    vectorize_symptoms,
    extract_symptoms_from_text,
    find_disease_row,
    similar_diseases,
    diseases_for_symptoms,
)


class MLPredictor:
    """Loads the trained models once and provides ``predict`` / ``lookup``."""

    def __init__(self, models_dir: Path = MODELS_DIR, csv_path: Path = DEFAULT_DATASET_PATH):
        self.models_dir = Path(models_dir)
        self.csv_path = Path(csv_path)
        self.dataset: Optional[pd.DataFrame] = None
        self.vocabulary: List[str] = []
        self.classifier = None
        self.regressor = None
        self.clusterer = None
        self.tfidf = None
        self.tfidf_matrix = None
        self.disease_names: List[str] = []
        self.cluster_map: Dict[int, List[str]] = {}
        self._loaded = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load(self) -> None:
        if self._loaded:
            return

        self.dataset = load_dataset(self.csv_path)
        self.vocabulary = self._load_vocabulary()

        self.classifier = joblib.load(self.models_dir / "disease_classifier.joblib")
        self.regressor = joblib.load(self.models_dir / "risk_regressor.joblib")
        self.clusterer = joblib.load(self.models_dir / "disease_clusterer.joblib")
        self.tfidf = joblib.load(self.models_dir / "symptom_tfidf.joblib")
        self.tfidf_matrix = joblib.load(self.models_dir / "symptom_tfidf_matrix.joblib")
        self.disease_names = joblib.load(self.models_dir / "disease_names.joblib")
        self.cluster_map = joblib.load(self.models_dir / "cluster_map.joblib")
        self._loaded = True

    def _load_vocabulary(self) -> List[str]:
        vocab_path = self.models_dir / "symptom_vocabulary.joblib"
        if vocab_path.exists():
            return joblib.load(vocab_path)
        # Fall back to a vocabulary derived from the dataset.
        return build_symptom_vocabulary(self.dataset)

    # ------------------------------------------------------------------
    # Symptom extraction + vectorisation
    # ------------------------------------------------------------------
    def extract_symptoms(self, text: str) -> List[str]:
        if not text:
            return []
        return extract_symptoms_from_text(text, self.vocabulary)

    def vectorize(self, symptoms: List[str]) -> np.ndarray:
        vec = vectorize_symptoms(symptoms, self.vocabulary)
        return np.asarray([vec], dtype=np.int8)

    # ------------------------------------------------------------------
    # Disease lookup
    # ------------------------------------------------------------------
    def get_disease_record(self, disease_name: str) -> Optional[Dict[str, Any]]:
        if self.dataset is None:
            return None
        row = find_disease_row(self.dataset, disease_name)
        if row is None:
            return None
        return row.to_dict()

    def search_disease(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if self.dataset is None:
            return []
        return similar_diseases(self.dataset, query, k=k)

    def get_diseases_for_symptoms(
        self, symptoms: List[str], k: int = 5
    ) -> List[Dict[str, Any]]:
        if self.dataset is None:
            return []
        return diseases_for_symptoms(self.dataset, symptoms, top_k=k)

    # ------------------------------------------------------------------
    # Risk + cluster lookup for a disease name
    # ------------------------------------------------------------------
    def risk_for_disease(self, disease_name: str) -> Optional[Dict[str, Any]]:
        record = self.get_disease_record(disease_name)
        if not record:
            return None
        return {
            "disease": record.get("disease"),
            "severity": record.get("severity"),
            "risk_score": int(record.get("risk_score", 0) or 0),
            "description": record.get("description", ""),
            "specialty": record.get("specialty", ""),
            "medications": list(record.get("medications", []) or []),
            "tests": list(record.get("tests", []) or []),
            "symptoms": list(record.get("symptoms", []) or []),
        }

    def cluster_for_disease(self, disease_name: str) -> Optional[Dict[str, Any]]:
        record = self.get_disease_record(disease_name)
        if not record:
            return None
        vec = self.vectorize(record.get("symptoms", []))
        cluster_id = int(self.clusterer.predict(vec)[0])
        return {
            "cluster_id": cluster_id,
            "related_diseases": [d for d in self.cluster_map.get(cluster_id, []) if d.lower() != disease_name.lower()],
        }

    # ------------------------------------------------------------------
    # Symptom-only inference
    # ------------------------------------------------------------------
    def predict_from_symptoms(
        self,
        symptoms: List[str],
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """Predict disease, risk score, and similar diseases from symptoms."""
        self.load()

        if not symptoms:
            return {
                "symptoms": [],
                "top_diseases": [],
                "predicted_risk_score": None,
                "risk_level": "Unknown",
            }

        vec = self.vectorize(symptoms)
        probs = self.classifier.predict_proba(vec)[0]
        classes = self.classifier.classes_

        ranked = sorted(zip(classes, probs), key=lambda item: item[1], reverse=True)
        top_diseases = []
        for disease, prob in ranked[:top_k]:
            record = self.get_disease_record(disease) or {}
            top_diseases.append({
                "disease": disease,
                "probability": round(float(prob), 4),
                "severity": record.get("severity", "Unknown"),
                "risk_score": int(record.get("risk_score", 0) or 0),
                "medications": list(record.get("medications", []) or [])[:5],
                "tests": list(record.get("tests", []) or [])[:5],
                "description": record.get("description", ""),
                "specialty": record.get("specialty", ""),
            })

        predicted_risk = float(self.regressor.predict(vec)[0])
        predicted_risk = max(0.0, min(100.0, predicted_risk))
        cluster_id = int(self.clusterer.predict(vec)[0])
        related = [d for d in self.cluster_map.get(cluster_id, [])]

        return {
            "symptoms": symptoms,
            "top_diseases": top_diseases,
            "predicted_risk_score": round(predicted_risk, 1),
            "risk_level": _risk_level(predicted_risk),
            "cluster_id": cluster_id,
            "cluster_related_diseases": related,
        }

    # ------------------------------------------------------------------
    # Free-text inference
    # ------------------------------------------------------------------
    def predict_from_text(
        self,
        text: str,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        symptoms = self.extract_symptoms(text)
        result = self.predict_from_symptoms(symptoms, top_k=top_k)
        result["source_text"] = text
        return result

    # ------------------------------------------------------------------
    # Disease-only inference (no symptoms provided)
    # ------------------------------------------------------------------
    def predict_from_disease(self, disease_name: str) -> Dict[str, Any]:
        record = self.risk_for_disease(disease_name)
        if not record:
            return {"found": False, "query": disease_name}
        cluster = self.cluster_for_disease(disease_name) or {}
        return {
            "found": True,
            "query": disease_name,
            "disease": record["disease"],
            "severity": record["severity"],
            "risk_score": record["risk_score"],
            "risk_level": _risk_level(record["risk_score"]),
            "description": record["description"],
            "specialty": record["specialty"],
            "medications": record["medications"],
            "tests": record["tests"],
            "symptoms": record["symptoms"],
            "cluster": cluster,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _risk_level(score: float) -> str:
    if score >= 80:
        return "Critical"
    if score >= 60:
        return "High"
    if score >= 35:
        return "Medium"
    if score > 0:
        return "Low"
    return "Unknown"
