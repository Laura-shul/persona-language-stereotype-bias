"""
hf_api_client.py

Thin wrapper around the Hugging Face Inference Providers API, using the
official `huggingface_hub` client library. This library automatically
handles Hugging Face's routing (router.huggingface.co), so the code keeps
working even if the underlying API endpoint changes again in the future.
No model weights are ever downloaded locally -> zero load-time risk.

NOTE: this client is used for the masking experiment (fill_mask) and for
one exploratory validation attempt (classify_bias). Free-form text
generation for the main Persona/Language experiment is NOT done through
this API client -- it is done locally via models/local_chat_model.py
(Qwen2.5-0.5B-Instruct), because HF's free third-party chat-generation
routing proved unreliable during this project (see report Methods
section). An earlier API-based chat_completion() method was removed from
this file for that reason, to avoid leaving unused/misleading code in the
final submission.

Usage:
    export HF_TOKEN="hf_xxx"   # or pass token directly
    client = HFInferenceClient(token=os.environ["HF_TOKEN"])
    result = client.fill_mask("The nurse said that [MASK] was tired.")
    result = client.classify_bias("Some response text here.")
"""

import os
from huggingface_hub import InferenceClient


class HFInferenceClient:
    # "hf-inference" is HF's own free serverless infrastructure for
    # classic pipeline tasks (fill-mask, text classification), which does
    # NOT draw from the paid, third-party "Inference Providers" monthly
    # credits. Full org-prefixed model ID required by the current API.
    MASK_MODEL = "google-bert/bert-base-uncased"

    def __init__(self, token: str | None = None):
        self.token = token or os.environ.get("HF_TOKEN")
        if not self.token:
            raise ValueError(
                "No Hugging Face token found. Set HF_TOKEN env var or pass token=... "
                "Get a free token at https://huggingface.co/settings/tokens"
            )
        self.mask_client = InferenceClient(provider="hf-inference", api_key=self.token)

    def fill_mask(self, text: str, top_k: int = 5) -> list[dict]:
        """text must contain the literal token [MASK]. Returns a list of
        dicts sorted by score, each with a 'token_str' key, to keep the
        calling code unchanged regardless of the underlying client version."""
        results = self.mask_client.fill_mask(text, model=self.MASK_MODEL)
        normalized = []
        for r in results[:top_k]:
            token_str = getattr(r, "token_str", "") or ""
            score = getattr(r, "score", None)
            normalized.append({"token_str": token_str, "score": score})
        return normalized

    def classify_bias(self, text: str) -> float | None:
        """Scores a response using a dedicated pretrained bias classifier
        (d4data/bias-detection-model), directly mirroring the course
        technique from L10.1 (bias-bert-classifier), which we deliberately
        had NOT used for the primary metric (we chose a transparent lexicon
        instead). Returns the probability assigned to the 'Biased' label,
        or None if the model call fails. NOTE: in practice this model was
        not available on the free hf-inference tier at the time of this
        project (BadRequestError: "Model not supported by provider
        hf-inference"); we kept this method and documented the failure
        honestly in the report rather than silently deleting the attempt."""
        if not text or not text.strip():
            return None
        try:
            results = self.mask_client.text_classification(
                text[:512], model="d4data/bias-detection-model"
            )
        except Exception:
            return None
        for r in results:
            label = getattr(r, "label", "") or (r.get("label", "") if isinstance(r, dict) else "")
            score = getattr(r, "score", None) or (r.get("score") if isinstance(r, dict) else None)
            if label and "bias" in label.lower() and "non" not in label.lower():
                return score
        # fallback: if labels are LABEL_0/LABEL_1 style, assume index 1 = biased
        if results:
            last = results[-1]
            return getattr(last, "score", None) or (last.get("score") if isinstance(last, dict) else None)
        return None


if __name__ == "__main__":
    # quick smoke test (requires HF_TOKEN to be set)
    client = HFInferenceClient()
    print("Testing fill_mask...")
    print(client.fill_mask("The nurse said that [MASK] was tired."))
    print("Testing classify_bias...")
    print(client.classify_bias("This is a typical example sentence."))