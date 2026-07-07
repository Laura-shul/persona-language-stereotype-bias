"""
correlate_annotations.py

After you have manually filled in the 'human_rating' column in
results/annotation_sample.csv, run this script to compute the correlation
between your human ratings and the automatic Stereotype Density Score
(SDS). A significant positive correlation is evidence that SDS captures a
real signal, not an artifact of the specific word list -- this is the
independent validation of the main metric.

Usage:
    python correlate_annotations.py
"""

import os
import pandas as pd
from scipy import stats

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANNOTATION_CSV = os.path.join(BASE, "results", "annotation_sample.csv")


def main():
    df = pd.read_csv(ANNOTATION_CSV)
    df["human_rating"] = pd.to_numeric(df["human_rating"], errors="coerce")
    df["stereotype_density"] = pd.to_numeric(df["stereotype_density"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["human_rating", "stereotype_density"])
    after = len(df)
    if after < before:
        print(f"Note: dropped {before - after} rows with missing rating or density "
              f"(e.g. unrated rows, or Japanese responses excluded from SDS).")

    if len(df) < 3:
        print("Not enough rated rows yet. Fill in more of the 'human_rating' column first.")
        return

    r_pearson, p_pearson = stats.pearsonr(df["human_rating"], df["stereotype_density"])
    r_spearman, p_spearman = stats.spearmanr(df["human_rating"], df["stereotype_density"])

    print("=" * 60)
    print(f"Validation sample size: {len(df)} human-rated responses")
    print("=" * 60)
    print(f"Pearson  correlation (SDS vs. human rating): r={r_pearson:.3f}, p={p_pearson:.4f}")
    print(f"Spearman correlation (SDS vs. human rating): r={r_spearman:.3f}, p={p_spearman:.4f}")

    if p_spearman < 0.05 and r_spearman > 0:
        print("\n=> SDS correlates significantly and positively with human judgment.")
        print("   This supports SDS as a meaningful (if imperfect) proxy metric.")
    else:
        print("\n=> No significant positive correlation was found at this sample size.")
        print("   Report this honestly: it is a real limitation of the SDS metric,")
        print("   not a reason to hide or omit the validation attempt.")


if __name__ == "__main__":
    main()
