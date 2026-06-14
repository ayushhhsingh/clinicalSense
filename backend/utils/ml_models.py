"""Train and persist the ML models used by the clinical-sense system.

This script reads the curated ``data/disease_dataset.csv`` file, augments it
with a few synthetic samples (so that subset-of-symptoms inference works)
and trains three models:

* **Multiclass classification** – Random Forest mapping ``(symptoms) → disease``.
* **Regression** – Gradient Boosting mapping ``(symptoms) → risk_score``.
* **Clustering** – KMeans grouping diseases by their symptom profiles.

Trained artefacts are persisted to the ``models/`` directory so the backend
can load them quickly at startup.
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Make the project importable when running this script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    mean_absolute_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from backend.utils.data_processor import (  # noqa: E402
    DEFAULT_DATASET_PATH,
    MODELS_DIR,
    build_symptom_vocabulary,
    load_dataset,
    vectorize_symptoms,
)


# ---------------------------------------------------------------------------
# Synthetic sample generation
# ---------------------------------------------------------------------------

def augment_samples(
    df: pd.DataFrame,
    vocabulary: List[str],
    samples_per_disease: int = 6,
    seed: int = 42,
) -> Tuple[np.ndarray, List[str], List[int]]:
    """Generate (X, disease, risk_score) training samples.

    For every disease we keep the full symptom set and add ``samples_per_disease``
    additional random subsets of the disease's symptoms so the classifier
    learns to generalise from incomplete inputs.
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    X_rows: List[List[int]] = []
    y_disease: List[str] = []
    y_risk: List[int] = []

    for _, row in df.iterrows():
        disease = str(row["disease"])
        risk_score = int(row.get("risk_score", 0) or 0)
        symptoms = [s for s in row["symptoms"] if s]

        # Always keep the full canonical symptom set.
        canonical = vectorize_symptoms(symptoms, vocabulary)
        X_rows.append(canonical)
        y_disease.append(disease)
        y_risk.append(risk_score)

        # Add randomised subsets for augmentation.
        if symptoms:
            n = len(symptoms)
            for _ in range(samples_per_disease):
                # Pick between 1 and all symptoms to simulate partial input.
                k = rng.randint(1, n)
                subset = rng.sample(symptoms, k)
                vec = vectorize_symptoms(subset, vocabulary)
                X_rows.append(vec)
                y_disease.append(disease)
                # Risk may vary slightly when fewer symptoms are present.
                jitter = rng.randint(-5, 5)
                y_risk.append(max(0, min(100, risk_score + jitter)))

    X = np.asarray(X_rows, dtype=np.int8)
    return X, y_disease, y_risk


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_models(
    csv_path: Path = DEFAULT_DATASET_PATH,
    models_dir: Path = MODELS_DIR,
    samples_per_disease: int = 6,
) -> Dict[str, object]:
    df = load_dataset(csv_path)
    vocabulary = build_symptom_vocabulary(df)
    print(f"Dataset: {len(df)} diseases, vocabulary={len(vocabulary)} symptoms")

    X, y_disease, y_risk = augment_samples(
        df, vocabulary, samples_per_disease=samples_per_disease
    )
    print(f"Training samples after augmentation: {X.shape[0]}")

    # ------------------------------------------------------------------
    # 1) Multiclass classification
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_disease, test_size=0.2, random_state=42, stratify=y_disease
    )
    classifier = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced",
    )
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[Classification] Random Forest accuracy: {acc:.3f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    # ------------------------------------------------------------------
    # 2) Regression – risk score from symptoms
    # ------------------------------------------------------------------
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X, np.asarray(y_risk, dtype=np.float32),
        test_size=0.2, random_state=42,
    )
    regressor = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
    )
    regressor.fit(Xr_train, yr_train)
    yr_pred = regressor.predict(Xr_test)
    mae = mean_absolute_error(yr_test, yr_pred)
    r2 = r2_score(yr_test, yr_pred)
    print(f"[Regression] Gradient Boosting MAE: {mae:.2f}, R^2: {r2:.3f}")

    # ------------------------------------------------------------------
    # 3) Clustering – group similar diseases
    # ------------------------------------------------------------------
    n_clusters = min(8, max(2, len(df) // 6))
    clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    disease_features = np.asarray(
        [vectorize_symptoms(row["symptoms"], vocabulary) for _, row in df.iterrows()],
        dtype=np.int8,
    )
    cluster_labels = clusterer.fit_predict(disease_features)
    if len(set(cluster_labels)) > 1:
        sil = silhouette_score(disease_features, cluster_labels)
    else:
        sil = float("nan")
    print(f"[Clustering] KMeans n_clusters={n_clusters}, silhouette={sil:.3f}")

    # Map cluster_id → list of diseases for quick lookup later.
    cluster_map: Dict[int, List[str]] = {}
    for label, (_, row) in zip(cluster_labels, df.iterrows()):
        cluster_map.setdefault(int(label), []).append(str(row["disease"]))

    # ------------------------------------------------------------------
    # 4) TF-IDF over symptom text for fuzzy "symptom text → disease" recall
    # ------------------------------------------------------------------
    symptom_docs = [" ".join(s.lower() for s in row["symptoms"]) for _, row in df.iterrows()]
    disease_names = df["disease"].tolist()
    tfidf = TfidfVectorizer(lowercase=True)
    tfidf_matrix = tfidf.fit_transform(symptom_docs)

    # ------------------------------------------------------------------
    # Persist artefacts
    # ------------------------------------------------------------------
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(classifier, models_dir / "disease_classifier.joblib")
    joblib.dump(regressor, models_dir / "risk_regressor.joblib")
    joblib.dump(clusterer, models_dir / "disease_clusterer.joblib")
    joblib.dump(tfidf, models_dir / "symptom_tfidf.joblib")
    joblib.dump(tfidf_matrix, models_dir / "symptom_tfidf_matrix.joblib")
    joblib.dump(disease_names, models_dir / "disease_names.joblib")
    joblib.dump(vocabulary, models_dir / "symptom_vocabulary.joblib")
    joblib.dump(cluster_map, models_dir / "cluster_map.joblib")

    metadata = {
        "n_diseases": int(len(df)),
        "vocabulary_size": int(len(vocabulary)),
        "training_samples": int(X.shape[0]),
        "classifier_accuracy": float(acc),
        "regression_mae": float(mae),
        "regression_r2": float(r2),
        "n_clusters": int(n_clusters),
        "silhouette": float(sil) if not np.isnan(sil) else None,
        "samples_per_disease": int(samples_per_disease),
        "model_files": [
            "disease_classifier.joblib",
            "risk_regressor.joblib",
            "disease_clusterer.joblib",
            "symptom_tfidf.joblib",
            "symptom_tfidf_matrix.joblib",
            "disease_names.joblib",
            "symptom_vocabulary.joblib",
            "cluster_map.joblib",
        ],
    }
    with open(models_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Persist the cleaned dataset so the API can answer CSV-grounded questions
    # (e.g. "what medication is recommended for X") without re-parsing.
    df.to_csv(models_dir / "dataset_clean.csv", index=False)

    print(f"\nSaved trained artefacts to: {models_dir}")
    print(json.dumps(metadata, indent=2))
    return metadata


if __name__ == "__main__":
    train_models()
