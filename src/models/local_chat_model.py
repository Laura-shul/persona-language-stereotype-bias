"""
local_chat_model.py

Local (CPU-friendly) text completion using a small open-weight model via
the `transformers` library.

Why local instead of the API: Hugging Face's free Inference Providers
routing for text-generation has proven unstable in practice (available
models change frequently, and free credits for third-party providers are
easily exhausted). Running a small model locally is more reliable and
fully reproducible. The model is downloaded once (~1GB) and cached by
`transformers`; every call after that runs fully offline and is fast,
since the model has only 0.5B parameters.
"""

from transformers import pipeline

_generator = None


def _get_generator():
    """Lazily loads and caches the local text-generation pipeline (singleton
    pattern), so the ~0.5B-parameter model is loaded into memory only once
    per process, not once per prompt."""
    global _generator
    if _generator is None:
        _generator = pipeline(
            "text-generation",
            model="Qwen/Qwen2.5-0.5B-Instruct",
            device=-1,  # force CPU; avoids any GPU/driver setup requirements
        )
    return _generator


def local_chat_completion(prompt: str, max_new_tokens: int = 100) -> str:
    """Generates one stochastic chat-style completion for the given prompt
    using the local Qwen2.5-0.5B-Instruct model (temperature=0.7, sampling
    enabled). Called once per repeat in run_experiments.py, so three calls
    with the same prompt yield three different completions, which is the
    basis for the project's repeated-sampling design."""
    generator = _get_generator()
    messages = [{"role": "user", "content": prompt}]
    output = generator(
        messages,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        pad_token_id=generator.tokenizer.eos_token_id,
    )
    generated = output[0]["generated_text"]
    if isinstance(generated, list):
        # chat-style output: list of role/content dicts, last one is the reply
        return generated[-1]["content"].strip()
    return str(generated).strip()


if __name__ == "__main__":
    print("Downloading/loading model (first run only takes a few minutes)...")
    print(local_chat_completion("Describe a typical family dinner in one sentence."))