"""
compute_semantic_scores.py

Computes the semantic_stereotype_score (see evaluation/semantic_similarity.py)
for every Cultural-domain response in results/raw_results.csv, and saves it
back as a new column. Unlike the word-level Stereotype Density Score (SDS),
this embedding-based method does NOT require word segmentation, so it can
score Japanese responses too -- providing a genuine additional check that
SDS structurally cannot provide, not just a redundant third metric.

Usage:
    python compute_semantic_scores.py
    (first run downloads a ~470MB multilingual embedding model, one time)
"""

import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.semantic_similarity import semantic_stereotype_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_CSV = os.path.join(BASE, "results", "raw_results.csv")


def main():
    df = pd.read_csv(RESULTS_CSV)
    cult_mask = df["domain"] == "cultural"

    print(f"Computing semantic scores for {cult_mask.sum()} cultural responses...")
    print("(First call loads the embedding model -- may take a minute.)")

    semantic_scores = []
    for i, row in df.iterrows():
        if row["domain"] == "cultural":
            text = str(row["output"]) if pd.notna(row["output"]) else ""
            score = semantic_stereotype_score(text)
            semantic_scores.append(score)
        else:
            semantic_scores.append("")
        if i % 50 == 0:
            print(f"  {i}/{len(df)} done...")

    df["semantic_score"] = semantic_scores
    df.to_csv(RESULTS_CSV, index=False)
    print(f"Saved semantic_score column to {RESULTS_CSV}")


if __name__ == "__main__":
    main()