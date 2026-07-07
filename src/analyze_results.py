"""
analyze_results.py

Loads results/raw_results.csv, computes the causal effect
(control vs. treatment) with a two-sample t-test, and produces
publication-ready figures for the report.

Usage:
    python analyze_results.py
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.power import TTestIndPower
import matplotlib.pyplot as plt
import seaborn as sns

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_CSV = os.path.join(BASE, "results", "raw_results.csv")
FIGURES_DIR = os.path.join(BASE, "figures")
SUMMARY_CSV = os.path.join(BASE, "results", "summary_stats.csv")

sns.set_theme(style="whitegrid", context="talk")


def analyze_masking(df: pd.DataFrame) -> pd.DataFrame:
    """RQ: does the masked LM's gender association differ by profession?
    (baseline bias measurement, no treatment/control here -> descriptive + ANOVA-style)"""
    mask_df = df[df["domain"] == "gender_profession"].copy()
    mask_df["gender_score"] = pd.to_numeric(mask_df["gender_score"], errors="coerce")

    summary = mask_df.groupby("profession")["gender_score"].agg(["mean", "std", "count"]).reset_index()
    summary = summary.sort_values("mean")

    # Plot: gender association by profession
    plt.figure(figsize=(10, 6))
    colors = ["#4C72B0" if v < 0 else "#DD8452" for v in summary["mean"]]
    plt.barh(summary["profession"], summary["mean"], color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Gender Association Score (negative = female-leaning, positive = male-leaning)")
    plt.title("Masked LM Gender Association by Profession")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig1_gender_association_by_profession.png"), dpi=200)
    plt.close()

    return summary


def power_analysis(effect_size: float, n_per_group: float, alpha: float = 0.05) -> dict:
    """Given the OBSERVED effect size and sample size, compute:
    1. The statistical power we actually had to detect that effect.
    2. The sample size per group that WOULD have been required to
       detect that same effect with 80% power.
    This turns 'our sample was too small' from a hand-wave into an exact,
    computed number, which is standard practice when reporting a null
    result (post-hoc power / required-N analysis)."""
    analysis = TTestIndPower()
    observed_power = analysis.solve_power(
        effect_size=abs(effect_size), nobs1=n_per_group, alpha=alpha, ratio=1.0
    )
    required_n = analysis.solve_power(
        effect_size=abs(effect_size), power=0.8, alpha=alpha, ratio=1.0
    )
    return {"observed_power": observed_power, "required_n_per_group_for_80pct_power": required_n}


def analyze_by_topic_and_country(df: pd.DataFrame) -> None:
    """Addresses the reviewer request for 'interaction analysis': does the
    Persona effect vary by topic or by country, rather than being a single
    uniform effect? Uses one-way ANOVA / Kruskal-Wallis across groups,
    which is appropriate given the small per-group sample sizes (a full
    two-way ANOVA would need substantially more data per cell to be
    reliable, per the power analysis in Section 3.8 of the report)."""
    cult_df = df[df["domain"] == "cultural"].copy()
    cult_df["stereotype_density"] = pd.to_numeric(cult_df["stereotype_density"], errors="coerce")
    persona_df = cult_df[cult_df["condition"] == "persona"].dropna(subset=["stereotype_density"])

    print("=" * 60)
    print("INTERACTION CHECK 1: Does the Persona effect vary by COUNTRY?")
    print("=" * 60)
    if "country" in persona_df.columns and persona_df["country"].nunique() > 1:
        groups = [g["stereotype_density"].values for _, g in persona_df.groupby("country")]
        by_country = persona_df.groupby("country")["stereotype_density"].agg(["mean", "std", "count"])
        print(by_country)
        if all(len(g) > 1 for g in groups):
            h_stat, p_val = stats.kruskal(*groups)
            print(f"\nKruskal-Wallis across countries: H={h_stat:.3f}, p={p_val:.4f}")
            print("=> Significant" if p_val < 0.05 else "=> Not significant",
                  "-- country does" if p_val < 0.05 else "-- no evidence country",
                  "significantly moderate the Persona effect at this sample size.")

    print()
    print("=" * 60)
    print("INTERACTION CHECK 2: Does stereotype density vary by TOPIC?")
    print("=" * 60)
    if "topic" in cult_df.columns:
        topic_df = cult_df.dropna(subset=["stereotype_density"])
        by_topic = topic_df.groupby("topic")["stereotype_density"].agg(["mean", "std", "count"]).sort_values("mean", ascending=False)
        print(by_topic)
        groups = [g["stereotype_density"].values for _, g in topic_df.groupby("topic")]
        if len(groups) > 1 and all(len(g) > 1 for g in groups):
            h_stat, p_val = stats.kruskal(*groups)
            print(f"\nKruskal-Wallis across topics: H={h_stat:.3f}, p={p_val:.4f}")
            print("=> Significant" if p_val < 0.05 else "=> Not significant",
                  "-- topic does" if p_val < 0.05 else "-- no evidence topic",
                  "significantly moderate stereotype density at this sample size.")

        by_topic.to_csv(os.path.join(BASE, "results", "density_by_topic.csv"))

        # Plot: mean stereotype density by topic, sorted descending
        plt.figure(figsize=(10, 6))
        by_topic_sorted = by_topic.sort_values("mean", ascending=True)
        colors = plt.cm.RdYlGn_r((by_topic_sorted["mean"] - by_topic_sorted["mean"].min()) /
                                   (by_topic_sorted["mean"].max() - by_topic_sorted["mean"].min() + 1e-9))
        plt.barh(by_topic_sorted.index, by_topic_sorted["mean"], color=colors)
        plt.xlabel("Mean Stereotype Density Score")
        plt.title("Stereotype Density by Question Topic (all conditions combined)")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "fig4_density_by_topic.png"), dpi=200, bbox_inches="tight")
        plt.close()


def _cohens_d(a: pd.Series, b: pd.Series) -> float:
    if len(a) > 1 and len(b) > 1:
        return (a.mean() - b.mean()) / np.sqrt((a.std() ** 2 + b.std() ** 2) / 2)
    return float("nan")


def analyze_persona_effect(df: pd.DataFrame) -> dict:
    """RQ: do Persona-prompting and Language-prompting causally increase
    Stereotype Density compared to a neutral Control prompt?
    Full P11 design: three groups (Control / Persona / Language) with
    two independent t-tests (Persona vs Control, Language vs Control)."""
    cult_df = df[df["domain"] == "cultural"].copy()
    cult_df["stereotype_density"] = pd.to_numeric(cult_df["stereotype_density"], errors="coerce")

    control = cult_df[cult_df["condition"] == "control"]["stereotype_density"].dropna()
    persona = cult_df[cult_df["condition"] == "persona"]["stereotype_density"].dropna()
    language = cult_df[cult_df["condition"] == "language"]["stereotype_density"].dropna()

    t_persona, p_persona = stats.ttest_ind(persona, control, equal_var=False)
    t_language, p_language = stats.ttest_ind(language, control, equal_var=False)
    d_persona = _cohens_d(persona, control)
    d_language = _cohens_d(language, control)

    # Plot: three-group distribution
    plt.figure(figsize=(10, 6.5))
    plot_df = pd.concat([
        pd.DataFrame({"condition": "Control", "stereotype_density": control}),
        pd.DataFrame({"condition": "Persona", "stereotype_density": persona}),
        pd.DataFrame({"condition": "Language", "stereotype_density": language}),
    ])
    sns.boxplot(data=plot_df, x="condition", y="stereotype_density", hue="condition",
                palette=["#4C72B0", "#DD8452", "#55A868"], legend=False,
                order=["Control", "Persona", "Language"])
    sns.stripplot(data=plot_df, x="condition", y="stereotype_density", color="black",
                  alpha=0.5, jitter=True, order=["Control", "Persona", "Language"])
    plt.ylabel("Stereotype Density Score")
    plt.title(
        "Effect of Prompting Strategy on Stereotype Density\n"
        f"Persona vs. Control: t={t_persona:.2f}, p={p_persona:.4f}, d={d_persona:.2f}\n"
        f"Language vs. Control: t={t_language:.2f}, p={p_language:.4f}, d={d_language:.2f}",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig2_persona_effect_boxplot.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # Plot: effect by country, all three conditions
    plt.figure(figsize=(9, 6))
    by_country = cult_df.groupby(["country", "condition"])["stereotype_density"].mean().reset_index()
    by_country["country"] = by_country["country"].fillna("Baseline (no country)")
    sns.barplot(data=by_country, x="country", y="stereotype_density", hue="condition",
                palette=["#4C72B0", "#DD8452", "#55A868"],
                hue_order=["control", "persona", "language"])
    plt.ylabel("Mean Stereotype Density Score")
    plt.title("Stereotype Density by Country and Prompting Condition")
    plt.figtext(
        0.5, -0.02,
        "Note: Japan's Language bar is omitted (word-level lexicon does not apply to non-spaced text; see Limitations).\n"
        "Nigeria has no Language bar by design (its language condition is English, identical to Control).",
        ha="center", fontsize=9, style="italic", color="#555555",
    )
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig3_stereotype_by_country.png"), dpi=200, bbox_inches="tight")
    plt.close()

    persona_power = power_analysis(d_persona, min(len(persona), len(control)))
    language_power = power_analysis(d_language, min(len(language), len(control)))

    return {
        "control_mean": control.mean(), "control_std": control.std(), "control_n": len(control),
        "persona_mean": persona.mean(), "persona_std": persona.std(), "persona_n": len(persona),
        "language_mean": language.mean(), "language_std": language.std(), "language_n": len(language),
        "t_persona_vs_control": t_persona, "p_persona_vs_control": p_persona, "cohens_d_persona": d_persona,
        "t_language_vs_control": t_language, "p_language_vs_control": p_language, "cohens_d_language": d_language,
        "power_persona": persona_power["observed_power"],
        "required_n_persona_80pct": persona_power["required_n_per_group_for_80pct_power"],
        "power_language": language_power["observed_power"],
        "required_n_language_80pct": language_power["required_n_per_group_for_80pct_power"],
    }


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    df = pd.read_csv(RESULTS_CSV)

    print("=" * 60)
    print("EXPERIMENT 1: Masking task - gender association by profession")
    print("=" * 60)
    mask_summary = analyze_masking(df)
    print(mask_summary.to_string(index=False))

    print()
    print("=" * 60)
    print("EXPERIMENT 2: Persona effect on stereotype density (causal)")
    print("=" * 60)
    persona_stats = analyze_persona_effect(df)
    for k, v in persona_stats.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    sig_persona = "SIGNIFICANT" if persona_stats["p_persona_vs_control"] < 0.05 else "not significant"
    sig_language = "SIGNIFICANT" if persona_stats["p_language_vs_control"] < 0.05 else "not significant"
    print(f"\n=> Persona treatment effect is {sig_persona} at alpha=0.05.")
    print(f"=> Language treatment effect is {sig_language} at alpha=0.05.")

    print()
    analyze_by_topic_and_country(df)

    # Save combined summary
    summary_row = {"experiment": "persona_causal_effect", **persona_stats}
    pd.DataFrame([summary_row]).to_csv(SUMMARY_CSV, index=False)
    mask_summary.to_csv(os.path.join(BASE, "results", "masking_summary.csv"), index=False)
    print(f"\nSaved figures to {FIGURES_DIR}/ and summaries to {os.path.dirname(SUMMARY_CSV)}/")


if __name__ == "__main__":
    main()
