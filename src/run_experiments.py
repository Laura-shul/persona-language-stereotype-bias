"""
run_experiments.py

Runs all 235 prompts (75 gender-profession masking + 160 cultural
persona/language/control) through the HF Inference API and local model,
scoring them with the lightweight lexicon-based metrics. Produces a CSV
in results/ ready for statistical analysis and plotting.

Usage:
    export HF_TOKEN="hf_xxx"
    python run_experiments.py
"""

import json
import os
import sys
import time
import csv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hf_api_client import HFInferenceClient
from models.local_chat_model import local_chat_completion
from evaluation.bias_lexicon import (
    score_mask_prediction,
    gender_association_score,
    stereotype_density_score,
)

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "prompts.json")
RESULTS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "raw_results.csv")


def run_masking_experiment(client: HFInferenceClient, rows: list[dict]) -> list[dict]:
    """For each masking prompt, get top-1 predicted token and score it."""
    results = []
    for row in rows:
        try:
            preds = client.fill_mask(row["text"], top_k=3)
            top_token = preds[0]["token_str"].strip() if preds else ""
            score = score_mask_prediction(top_token)
        except Exception as e:
            print(f"  [WARN] mask prompt failed: {row['text'][:50]}... ({e})")
            top_token, score = "", 0
        results.append({
            **row,
            "output": top_token,
            "gender_score": score,
            "stereotype_density": None,
        })
        time.sleep(0.3)  # be polite to the free-tier API
    return results


def run_chat_experiment(rows: list[dict], n_repeats: int = 3) -> list[dict]:
    """For each cultural prompt, generate multiple stochastic completions
    (the local model samples with temperature=0.7, so repeated calls yield
    different text) and score each one. Repeating each prompt is standard
    practice for LLM evaluation, since a single sample is a noisy estimate
    of the model's typical behavior; it also balances the sample sizes
    across conditions for the statistical comparison."""
    results = []
    for row in rows:
        for repeat_idx in range(n_repeats):
            try:
                output = local_chat_completion(row["text"], max_new_tokens=100)
            except Exception as e:
                print(f"  [WARN] chat prompt failed: {row['text'][:50]}... ({e})")
                output = ""
            lang = infer_language(row)
            density = stereotype_density_score(output, lang=lang)
            results.append({
                **row,
                "repeat": repeat_idx,
                "output": output,
                "gender_score": None,
                "stereotype_density": "" if density is None else density,
            })
    return results


# Maps each persona/language country to the language the "language"-condition
# prompt was actually written in (control/persona prompts are always English).
COUNTRY_TO_LANG = {
    "Italy": "it",
    "Kazakhstan": "ru",
    "Japan": "ja",     # unsupported by our word-level lexicon -> excluded
    "Nigeria": "en",   # kept English by design
}


def infer_language(row: dict) -> str:
    if row["domain"] != "cultural" or row["condition"] in ("control", "persona"):
        return "en"
    return COUNTRY_TO_LANG.get(row["country"], "en")


def main():
    with open(DATA_PATH, encoding="utf-8") as f:
        all_prompts = json.load(f)

    mask_rows = [r for r in all_prompts if r["domain"] == "gender_profession"]
    chat_rows = [r for r in all_prompts if r["domain"] == "cultural"]  # includes control, persona, AND language conditions (full P11 design)

    print(f"Masking experiment: {len(mask_rows)} prompts")
    print(f"Chat completion experiment: {len(chat_rows)} prompts")

    client = HFInferenceClient()

    print("Running masking experiment...")
    mask_results = run_masking_experiment(client, mask_rows)

    print("Running chat completion experiment (local model, first run downloads ~1GB)...")
    chat_results = run_chat_experiment(chat_rows)

    all_results = mask_results + chat_results

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    fieldnames = ["domain", "template_id", "topic", "condition", "profession", "country", "repeat",
                  "text", "output", "gender_score", "stereotype_density"]
    with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"Saved {len(all_results)} results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
