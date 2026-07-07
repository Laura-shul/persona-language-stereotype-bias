"""
compute_classifier_scores.py

Scores every Cultural-domain response with a dedicated, pretrained bias
classifier (d4data/bias-detection-model) via the free HF Inference API.
This directly mirrors the course technique from Lecture 10.1
(bias-bert-classifier).

Usage:
    export HF_TOKEN="hf_xxx"
    python compute_classifier_scores.py
"""

import os
import sys
import time
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hf_api_client import HFInferenceClient

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_CSV = os.path.join(BASE, "results", "raw_results.csv")


def main():
    df = pd.read_csv(RESULTS_CSV)
    cult_mask = df["domain"] == "cultural"
    n_total = cult_mask.sum()
    print(f"Scoring {n_total} cultural responses with d4data/bias-detection-model...")

    client = HFInferenceClient()

    classifier_scores = []
    failures = 0
    for i, row in df.iterrows():
        if row["domain"] == "cultural":
            text = str(row["output"]) if pd.notna(row["output"]) else ""
            score = client.classify_bias(text)
            if score is None:
                failures += 1
            classifier_scores.append(score if score is not None else "")
            time.sleep(0.3)
        else:
            classifier_scores.append("")
        if i % 50 == 0:
            print(f"  {i}/{len(df)} done... ({failures} failures so far)")

    df["classifier_bias_score"] = classifier_scores
    df.to_csv(RESULTS_CSV, index=False)
    print(f"Saved classifier_bias_score column to {RESULTS_CSV}")
    print(f"Total failures (skipped/empty): {failures}/{n_total}")


if __name__ == "__main__":
    main()