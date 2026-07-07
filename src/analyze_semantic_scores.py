"""
analyze_semantic_scores.py

Two analyses using the embedding-based semantic_score computed by
compute_semantic_scores.py:

1. Correlation between semantic_score and the lexicon-based SDS, across
   the full dataset (a much larger and more powerful check than the
   45-response LLM-judge sample).
2. Independent replication of the Persona-vs-Control and
   Language-vs-Control t-tests using semantic_score as the outcome
   instead of SDS -- and, uniquely, INCLUDING Japanese responses this
   time, since the embedding method does not require word segmentation.

Usage:
    python analyze_semantic_scores.py
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_CSV = os.path.join(BASE, "results", "raw_results.csv")
FIGURES_DIR = os.path.join(BASE, "figures")


def cohens_d(a, b):
    if len(a) > 1 and len(b) > 1:
        return (a.mean() - b.mean()) / np.sqrt((a.std() ** 2 + b.std() ** 2) / 2)
    return float("nan")


def main():
    df = pd.read_csv(RESULTS_CSV)
    cult_df = df[df["domain"] == "cultural"].copy()
    cult_df["stereotype_density"] = pd.to_numeric(cult_df["stereotype_density"], errors="coerce")
    cult_df["semantic_score"] = pd.to_numeric(cult_df["semantic_score"], errors="coerce")

    print("=" * 60)
    print("CORRELATION: semantic_score vs. SDS (full dataset)")
    print("=" * 60)
    both = cult_df.dropna(subset=["stereotype_density", "semantic_score"])
    r_p, p_p = stats.pearsonr(both["semantic_score"], both["stereotype_density"])
    r_s, p_s = stats.spearmanr(both["semantic_score"], both["stereotype_density"])
    print(f"n={len(both)}")
    print(f"Pearson:  r={r_p:.3f}, p={p_p:.4f}")
    print(f"Spearman: r={r_s:.3f}, p={p_s:.4f}")

    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=both, x="stereotype_density", y="semantic_score",
                     hue="condition", palette=["#4C72B0", "#DD8452", "#55A868"], alpha=0.6)
    plt.xlabel("Stereotype Density Score (lexicon-based)")
    plt.ylabel("Semantic Similarity Score (embedding-based)")
    plt.title(f"SDS vs. Semantic Score (n={len(both)})\nPearson r={r_p:.3f}, p={p_p:.4f}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig5_sds_vs_semantic_scatter.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved scatter plot to {FIGURES_DIR}/fig5_sds_vs_semantic_scatter.png")

    print()
    print("=" * 60)
    print("INDEPENDENT REPLICATION using semantic_score (includes Japan!)")
    print("=" * 60)
    control = cult_df[cult_df["condition"] == "control"]["semantic_score"].dropna()
    persona = cult_df[cult_df["condition"] == "persona"]["semantic_score"].dropna()
    language = cult_df[cult_df["condition"] == "language"]["semantic_score"].dropna()

    t_p, p_val_p = stats.ttest_ind(persona, control, equal_var=False)
    t_l, p_val_l = stats.ttest_ind(language, control, equal_var=False)
    d_p = cohens_d(persona, control)
    d_l = cohens_d(language, control)

    print(f"Control:  mean={control.mean():.4f}, n={len(control)}")
    print(f"Persona:  mean={persona.mean():.4f}, n={len(persona)}, t={t_p:.2f}, p={p_val_p:.4f}, d={d_p:.2f}")
    print(f"Language: mean={language.mean():.4f}, n={len(language)} (incl. Japan), t={t_l:.2f}, p={p_val_l:.4f}, d={d_l:.2f}")

    print()
    sig_p = "SIGNIFICANT" if p_val_p < 0.05 else "not significant"
    sig_l = "SIGNIFICANT" if p_val_l < 0.05 else "not significant"
    print(f"=> Persona effect (semantic_score) is {sig_p}.")
    print(f"=> Language effect (semantic_score) is {sig_l}.")

    # By-country breakdown including Japan for language
    print()
    print("Language condition by country (semantic_score, now includes Japan):")
    lang_df = cult_df[cult_df["condition"] == "language"]
    print(lang_df.groupby("country")["semantic_score"].agg(["mean", "std", "count"]))


if __name__ == "__main__":
    main()