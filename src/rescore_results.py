"""
rescore_results.py

Re-scores the already-generated outputs in results/raw_results.csv using
the language-aware stereotype lexicon, WITHOUT re-running the (slow) local
generation model. Run this once after updating bias_lexicon.py.

Usage:
    python rescore_results.py
"""

import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.bias_lexicon import stereotype_density_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_CSV = os.path.join(BASE, "results", "raw_results.csv")

# Maps each country used in the "persona"/"language" conditions to the
# language its "language"-condition prompt was actually written in.
COUNTRY_TO_LANG = {
    "Italy": "it",
    "Kazakhstan": "ru",
    "Japan": "ja",       # unsupported -> will be scored as None (excluded)
    "Nigeria": "en",     # kept English by design (see prompt_builder.py)
}


def infer_lang(row) -> str:
    if row["domain"] != "cultural":
        return "en"
    if row["condition"] in ("control", "persona"):
        # control and persona prompts are always written in English
        return "en"
    # condition == "language": the actual language depends on the country
    return COUNTRY_TO_LANG.get(row["country"], "en")


def main():
    df = pd.read_csv(RESULTS_CSV)

    new_density = []
    n_excluded = 0
    for _, row in df.iterrows():
        if row["domain"] != "cultural":
            new_density.append("")
            continue
        lang = infer_lang(row)
        if row["condition"] == "language" and lang == "ja":
            n_excluded += 1
        output_text = str(row["output"]) if pd.notna(row["output"]) else ""
        score = stereotype_density_score(output_text, lang=lang)
        new_density.append("" if score is None else score)

    df["stereotype_density"] = new_density
    df.to_csv(RESULTS_CSV, index=False)

    print(f"Re-scored {len(df)} rows using language-aware lexicons.")
    print(f"Excluded (unsupported language, e.g. Japanese): {n_excluded} rows -> stereotype_density left blank.")
    print(f"Saved back to {RESULTS_CSV}")


if __name__ == "__main__":
    main()