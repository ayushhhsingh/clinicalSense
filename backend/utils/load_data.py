"""Legacy entry point – kept for backwards compatibility.

New code should import from :mod:`backend.utils.data_processor`. This module
exists so the previous ``python backend/utils/load_data.py`` command still
loads the curated dataset from the local CSV file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from backend.utils.data_processor import (
    DEFAULT_DATASET_PATH,
    build_symptom_vocabulary,
    load_dataset,
    vectorize_symptoms,
)


def main(csv_path: Path = DEFAULT_DATASET_PATH, output: Path = None) -> int:
    if not csv_path.exists():
        print("ERROR: Dataset not found at", csv_path)
        print("Place the curated CSV at", csv_path)
        return 1

    df = load_dataset(csv_path)
    vocab = build_symptom_vocabulary(df)
    print(f"Loaded {len(df)} diseases | vocabulary={len(vocab)} symptoms")

    rows: List[dict] = []
    for _, row in df.iterrows():
        rows.append({
            "patient_id": f"P{_:05d}",
            "note_text": " | ".join(row["symptoms"]),
            "medical_specialty": row.get("specialty", ""),
            "disease": row["disease"],
            "severity": row["severity"],
            "risk_score": int(row.get("risk_score", 0) or 0),
            "medications": row["medications"],
            "tests": row["tests"],
        })

    if output is None:
        output = Path("clinical_notes.json")
    with open(output, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"Saved {len(rows)} records to {output}")
    return 0


if __name__ == "__main__":
    import sys

    exit(main())
