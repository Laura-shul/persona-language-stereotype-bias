"""
analyze_classifier_scores.py

Final validation check: correlates the pretrained bias-classifier score
(d4data/bias-detection-model, the exact technique from course Lecture 10.1)
with SDS, and independently replicates the core comparisons.

Usage:
    python analyze_classifier_scores.py
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
    cult_df["classifier_bias_score"] = pd.to_numeric(cult_df["classifier_bias_score"], errors="coerce")

    print("=" * 60)
    print("CORRELATION: classifier_bias_score vs. SDS")
    print("=" * 60)
    both = cult_df.dropna(subset=["stereotype_density", "classifier_bias_score"])

    if len(both) < 2:
        print(
            "No valid classifier_bias_score values found (n="
            f"{len(both)}). This is expected: the d4data/bias-detection-model\n"
            "was not available on the free Hugging Face hf-inference tier at\n"
            "the time of this project (BadRequestError: 'Model not supported\n"
            "by provider hf-inference'). This exploratory validation attempt\n"
            "is documented as a known limitation in the report rather than\n"
            "silently omitted. Run compute_classifier_scores.py first if a\n"
            "working model endpoint becomes available."
        )
        return

    r_p, p_p = stats.pearsonr(both["classifier_bias_score"], both["stereotype_density"])
    r_s, p_s = stats.spearmanr(both["classifier_bias_score"], both["stereotype_density"])
    print(f"n={len(both)}")
    print(f"Pearson:  r={r_p:.3f}, p={p_p:.4f}")
    print(f"Spearman: r={r_s:.3f}, p={p_s:.4f}")

    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=both, x="stereotype_density", y="classifier_bias_score",
                     hue="condition", palette=["#4C72B0", "#DD8452", "#55A868"], alpha=0.6)
    plt.xlabel("Stereotype Density Score (lexicon-based)")
    plt.ylabel("Bias Classifier Score (d4data/bias-detection-model)")
    plt.title(f"SDS vs. Bias Classifier (n={len(both)})\nPearson r={r_p:.3f}, p={p_p:.4f}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig6_sds_vs_classifier_scatter.png"), dpi=200, bbox_inches="tight")
    plt.close()

    print()
    print("=" * 60)
    print("INDEPENDENT REPLICATION using classifier_bias_score (incl. Japan!)")
    print("=" * 60)
    control = cult_df[cult_df["condition"] == "control"]["classifier_bias_score"].dropna()
    persona = cult_df[cult_df["condition"] == "persona"]["classifier_bias_score"].dropna()
    language = cult_df[cult_df["condition"] == "language"]["classifier_bias_score"].dropna()

    t_p, p_val_p = stats.ttest_ind(persona, control, equal_var=False)
    t_l, p_val_l = stats.ttest_ind(language, control, equal_var=False)
    d_p = cohens_d(persona, control)
    d_l = cohens_d(language, control)

    print(f"Control:  mean={control.mean():.4f}, n={len(control)}")
    print(f"Persona:  mean={persona.mean():.4f}, n={len(persona)}, t={t_p:.2f}, p={p_val_p:.4f}, d={d_p:.2f}")
    print(f"Language: mean={language.mean():.4f}, n={len(language)} (incl. Japan), t={t_l:.2f}, p={p_val_l:.4f}, d={d_l:.2f}")

    sig_p = "SIGNIFICANT" if p_val_p < 0.05 else "not significant"
    sig_l = "SIGNIFICANT" if p_val_l < 0.05 else "not significant"
    print(f"\n=> Persona effect (classifier) is {sig_p}.")
    print(f"=> Language effect (classifier) is {sig_l}.")


if __name__ == "__main__":
    main()