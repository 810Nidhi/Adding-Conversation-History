import config

from huggingface_hub import InferenceClient

MODELS = getattr(
    config,
    "HF_MODELS",
    ["meta-llama/Llama-3.1-8B-Instruct"],
)


def generate_response(
    prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 512,
) -> str:
    key = getattr(config, "HF_API_KEY", None)

    if not key:
        return "Error: HF_API_KEY missing in config.py"

    last_err = None

    for model in MODELS:
        try:
            client = InferenceClient(model=model, token=key)
            response = client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            last_err = e
            print(f"Model failed: {model}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error details: {e}")

    if last_err is not None:
        return (
            "Hugging Face API Error!\n"
            f"Error type: {type(last_err).__name__}\n"
            f"Details: {last_err}"
        )

    return "Hugging Face API Error!"