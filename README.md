# Persona vs. Language: A Causal Analysis of Prompting Strategies on Stereotype Expression in LLMs

**Course:** Natural Language Processing — Università degli Studi di Milano
**Student:** Laura Shulbayeva

## Research Question

To what extent do explicit persona-based prompts and native-language prompts causally
shift the expression of stereotypes in LLM outputs, compared to a neutral baseline prompt?

## Causal Framework

- **Outcome variable:** Stereotype Density Score (SDS) — the fraction of output tokens
  belonging to a curated lexicon of stereotype-associated terms.
- **Treatment:**
  - *Persona*: prompt explicitly assigns a cultural persona (e.g., "As a Nigerian
    person, answer: ...").
  - *Language*: the same question, written directly in a language associated with
    the target culture (Italian, Russian, Japanese).
- **Control:** neutral prompt, with no persona or language cue.
- **Total effect:** difference in mean SDS between each treatment group and control,
  tested with independent two-sample t-tests.

## Summary of Findings

Both Persona and Language prompting produce a statistically significant change in
stereotype density relative to control (p=0.034 and p=0.011 respectively), in opposite
directions: Persona prompting increases stereotype density, while Language prompting
decreases it. A country-level and topic-level interaction analysis further shows that
the magnitude of the Persona effect varies significantly across the four cultural
contexts studied, and that stereotype density is strongly topic-dependent.

To assess the robustness of this finding, the same comparison was independently
replicated using three further validation approaches (four checks in total, since
human annotation was conducted in two separate rounds): two rounds of human
annotation by the author, an LLM-as-a-Judge rating, and an embedding-based semantic
similarity score. Three of these four checks (both human-annotation rounds and the
LLM-judge rating) found no significant correlation with SDS; the semantic score
correlated significantly but only moderately with SDS, and did not replicate the
main causal effect when substituted for SDS in the core comparison. This indicates
that the causal conclusion is, at present, specific to the lexicon-based
operationalization of stereotype density rather than a fully metric-independent
effect. This cross-metric analysis, and its implications, is discussed in detail in
`report.pdf`.

## Metrics

| # | Metric | Implementation |
|---|---|---|
| 1 | Stereotype Density Score (SDS) — primary metric | `src/evaluation/bias_lexicon.py` |
| 2 | Gender Association Score (masking experiment) | `src/evaluation/bias_lexicon.py` |
| 3 | Human annotation (two rounds: intuitive, then checklist-based) | `src/prepare_annotation_sample.py`, `src/correlate_annotations.py` |
| 4 | LLM-as-a-Judge rating | `src/correlate_annotations.py` |
| 5 | Embedding-based semantic similarity | `src/evaluation/semantic_similarity.py` |

A sixth validation attempt, using a pretrained bias classifier mirroring the course's
own classification technique, is implemented in `src/compute_classifier_scores.py`;
this model was not available on the free Hugging Face inference tier at the time of
the experiments, and this limitation is documented in `report.pdf`.

## Models

- **Masking experiment:** `google-bert/bert-base-uncased`, via the Hugging Face
  Inference API.
- **Persona/Language chat completion:** `Qwen/Qwen2.5-0.5B-Instruct`, run locally via
  `transformers`, ensuring full reproducibility independent of third-party API
  availability.
- **Semantic similarity:** `paraphrase-multilingual-MiniLM-L12-v2`, run locally.

## Dataset

- 20 everyday-life questions spanning 11 thematic categories (family, religion, work,
  education, celebration, social norms, and others).
- 4 cultural contexts: Nigeria, Italy, Kazakhstan, Japan.
- 15 professions (gender-profession masking experiment).
- 555 total observations (75 masking + 480 cultural, with 3 stochastic completions per
  cultural prompt).

Prompts were custom-designed for this project rather than sourced from an existing
benchmark; the rationale for this choice is discussed in `report.pdf`, Section 2.3.

## Repository Structure

```
nlp-project/
├── report.pdf                        # final report (root copy, per course requirement)
├── report/
│   ├── report.tex
│   ├── report.pdf                    # duplicate, needed for LaTeX compilation
│   └── figures/                      # copies of figures used by LaTeX
├── src/
│   ├── data_gen/
│   │   └── prompt_builder.py         # generates the experimental prompt set
│   ├── models/
│   │   ├── hf_api_client.py          # Hugging Face Inference API wrapper
│   │   └── local_chat_model.py       # local Qwen2.5-0.5B-Instruct wrapper
│   ├── evaluation/
│   │   ├── bias_lexicon.py           # SDS and Gender Association Score
│   │   └── semantic_similarity.py    # embedding-based validation metric
│   ├── run_experiments.py            # runs the full experiment
│   ├── analyze_results.py            # main statistical analysis and figures
│   ├── error_analysis.py             # response categorization and qualitative examples
│   ├── prepare_annotation_sample.py  # builds the human-annotation sample
│   ├── correlate_annotations.py      # validates SDS against human/LLM ratings
│   ├── compute_semantic_scores.py    # computes the semantic similarity metric
│   ├── analyze_semantic_scores.py    # cross-metric validation analysis
│   ├── compute_classifier_scores.py  # bias-classifier validation attempt
│   ├── analyze_classifier_scores.py  # bias-classifier validation analysis
│   └── rescore_results.py            # utility for re-scoring existing outputs
├── tests/
│   └── test_bias_lexicon.py          # unit tests for the core scoring functions
├── notebooks/
│   └── demo.ipynb                    # demonstration of the pipeline components
├── data/
│   └── prompts.json                  # generated prompt set
├── results/                          # experiment outputs (CSVs)
├── figures/                          # report figures (PNGs)
├── slides/                           # presentation slides
├── .gitignore
├── requirements.txt
└── README.md
```

## Reproducing the Experiments

```bash
pip install -r requirements.txt

# Generate the prompt set
python src/data_gen/prompt_builder.py

# Run the full experiment (requires an HF_TOKEN environment variable)
export HF_TOKEN="hf_xxx"
python src/run_experiments.py

# Main statistical analysis and figures
python src/analyze_results.py

# Response categorization and qualitative examples
python src/error_analysis.py

# Semantic-score validation
python src/compute_semantic_scores.py
python src/analyze_semantic_scores.py

# Human-annotation / LLM-judge validation
# (results/annotation_sample.csv already contains the author's real ratings;
# prepare_annotation_sample.py is intentionally not re-run here, since it
# would prompt for confirmation before overwriting them)
python src/correlate_annotations.py
```

`notebooks/demo.ipynb` provides a shorter, illustrative walkthrough of the core
pipeline components, running directly on the already-generated results.

## AI Usage Disclaimer

This project was developed with the assistance of Claude (Anthropic). The full
disclaimer, including the specific use of Claude as an LLM judge for one of the
validation checks, is provided in `report.pdf`. The author directed the project
throughout, made all key methodological decisions, and personally executed every
experimental run and manual step; Claude assisted with code scaffolding, debugging,
statistical methodology guidance, and drafting of descriptive text.
