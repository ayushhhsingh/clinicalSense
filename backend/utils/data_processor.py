"""Data loading utilities for the clinical-sense project.

Loads the disease/symptom/medication dataset from CSV, normalises it, and
exposes helper functions for downstream ML + agents.
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# Default project paths -------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DATASET_PATH = DATA_DIR / "disease_dataset.csv"
MODELS_DIR = PROJECT_ROOT / "models"


# Canonical symptom vocabulary -----------------------------------------------
# Built dynamically from the dataset the first time ``load_dataset`` runs, but
# a sane minimum is provided so that downstream code can import it safely even
# when the dataset is unavailable (e.g. unit tests, cold start, etc).

FALLBACK_SYMPTOMS: List[str] = [
    "fever", "chills", "fatigue", "weakness", "weight loss", "weight gain",
    "night sweats", "loss of appetite",
    "headache", "severe headache", "migraine", "dizziness", "vertigo",
    "confusion", "memory loss", "seizures", "tremor", "numbness", "tingling",
    "blurred vision", "vision problems", "loss of vision", "eye pain",
    "sore throat", "hoarseness", "cough", "chronic cough", "coughing blood",
    "shortness of breath", "wheezing", "chest tightness", "chest pain",
    "palpitations", "rapid heartbeat", "irregular heartbeat",
    "abdominal pain", "nausea", "vomiting", "diarrhea", "constipation",
    "bloody diarrhea", "blood in stool", "blood in urine", "painful urination",
    "frequent urination", "excessive thirst", "joint pain", "joint swelling",
    "muscle pain", "back pain", "neck stiffness", "rash", "skin rash",
    "itching", "red skin", "jaundice", "swelling", "leg swelling",
    "swollen lymph nodes", "anxiety", "depression", "insomnia", "sleep problems",
]


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _split_pipe(value: Any) -> List[str]:
    """Split a pipe-delimited cell into a cleaned list of strings."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value)
    if not text or text.lower() == "nan":
        return []
    return [chunk.strip() for chunk in text.split("|") if chunk.strip()]


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def load_dataset(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """Load the disease dataset and apply schema normalisation.

    The returned frame has these columns:

    ``disease, severity, risk_score, specialty, description, symptoms,
    medications, tests, symptom_count, med_count``
    """
    path = Path(csv_path) if csv_path else DEFAULT_DATASET_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Disease dataset not found at {path}. "
            f"Place the CSV file under {DATA_DIR}/."
        )

    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"disease", "symptoms", "severity", "risk_score",
                "medications", "tests"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )

    df["disease"] = df["disease"].astype(str).str.strip()
    df["severity"] = df["severity"].astype(str).str.strip()
    df["risk_score"] = df["risk_score"].apply(_coerce_int)
    df["symptoms"] = df["symptoms"].apply(_split_pipe)
    df["medications"] = df["medications"].apply(_split_pipe)
    df["tests"] = df["tests"].apply(_split_pipe)
    df["specialty"] = df.get("specialty", "").apply(_coerce_text)
    df["description"] = df.get("description", "").apply(_coerce_text)
    df["symptom_count"] = df["symptoms"].apply(len)
    df["med_count"] = df["medications"].apply(len)
    return df


# ---------------------------------------------------------------------------
# Symptom vocabulary helpers
# ---------------------------------------------------------------------------

def build_symptom_vocabulary(df: pd.DataFrame) -> List[str]:
    """Build a sorted, de-duplicated symptom vocabulary from the dataset."""
    vocab: set = set()
    for symptoms in df["symptoms"]:
        vocab.update(s.strip().lower() for s in symptoms if s.strip())
    # Make sure canonical fallbacks are present.
    vocab.update(s.lower() for s in FALLBACK_SYMPTOMS)
    return sorted(vocab)


def vectorize_symptoms(symptoms: List[str], vocabulary: List[str]) -> List[int]:
    """One-hot encode ``symptoms`` against ``vocabulary`` (multi-hot)."""
    sym_set = {s.strip().lower() for s in symptoms if s.strip()}
    return [1 if s.lower() in sym_set else 0 for s in vocabulary]


def extract_symptoms_from_text(text: str, vocabulary: List[str]) -> List[str]:
    """Return the symptoms from ``vocabulary`` mentioned inside ``text``."""
    if not text:
        return []
    text_lower = text.lower()
    found: List[str] = []
    for symptom in vocabulary:
        if not symptom:
            continue
        # Use word-ish boundaries to avoid partial matches inside other words.
        pattern = r"\b" + re.escape(symptom) + r"\b"
        if re.search(pattern, text_lower):
            found.append(symptom)
    return found


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def find_disease_row(df: pd.DataFrame, query: str) -> Optional[pd.Series]:
    """Find a disease row by case-insensitive name (exact or contains)."""
    if not query:
        return None
    q = query.strip().lower()
    if not q:
        return None
    for _, row in df.iterrows():
        name = str(row["disease"]).lower()
        if name == q or q in name or name in q:
            return row
    return None


def similar_diseases(df: pd.DataFrame, query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Return up to ``k`` diseases whose names resemble ``query``."""
    q = (query or "").strip().lower()
    if not q:
        return []
    scored: List[Tuple[float, pd.Series]] = []
    for _, row in df.iterrows():
        name = str(row["disease"]).lower()
        score = _similarity_score(q, name)
        if score > 0:
            scored.append((score, row))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [r.to_dict() for _, r in scored[:k]]


def _similarity_score(a: str, b: str) -> float:
    """Cheap character-overlap similarity, good enough for short labels."""
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.8
    a_tokens = set(a.split())
    b_tokens = set(b.split())
    if not a_tokens or not b_tokens:
        return 0.0
    overlap = a_tokens & b_tokens
    if overlap:
        return 0.6 * len(overlap) / max(len(a_tokens), len(b_tokens))
    # Trigram overlap as a final fallback.
    a_trigrams = {a[i:i + 3] for i in range(len(a) - 2)} or {a}
    b_trigrams = {b[i:i + 3] for i in range(len(b) - 2)} or {b}
    tri_overlap = a_trigrams & b_trigrams
    return 0.4 * len(tri_overlap) / max(len(a_trigrams), len(b_trigrams))


def diseases_for_symptoms(
    df: pd.DataFrame, symptoms: List[str], top_k: int = 5
) -> List[Dict[str, Any]]:
    """Return diseases ranked by overlap with the supplied symptoms."""
    if not symptoms:
        return []
    target = {s.strip().lower() for s in symptoms if s.strip()}
    rows: List[Tuple[int, pd.Series]] = []
    for _, row in df.iterrows():
        disease_symptoms = {s.lower() for s in row["symptoms"]}
        overlap = len(target & disease_symptoms)
        if overlap:
            rows.append((overlap, row))
    rows.sort(key=lambda item: (item[0], item[1].get("risk_score", 0)), reverse=True)
    return [r.to_dict() for _, r in rows[:top_k]]


# ---------------------------------------------------------------------------
# Public convenience entry point
# ---------------------------------------------------------------------------

def load_full(csv_path: Optional[Path] = None) -> Tuple[pd.DataFrame, List[str]]:
    """Return ``(dataset, symptom_vocabulary)`` ready for ML use."""
    df = load_dataset(csv_path)
    vocab = build_symptom_vocabulary(df)
    return df, vocab


if __name__ == "__main__":
    df, vocab = load_full()
    print(f"Loaded {len(df)} disease records")
    print(f"Symptom vocabulary size: {len(vocab)}")
    print("Severity distribution:")
    print(df["severity"].value_counts())
