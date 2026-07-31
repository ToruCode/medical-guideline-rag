"""OpenAI Chat Completions-based implementation of the Llm port.

Importing openai is deferred to __init__ (not done at module import
time), so merely referencing the OpenAiLlm class - as
app/api/dependencies.py always does, regardless of which llm_provider
is configured - never pays the cost of importing the SDK. See
docs/adr/0011-real-embedding-and-llm-adapters.md.

No exception raised by the OpenAI client is caught here; it propagates
to the caller unchanged, matching GenerateAnswerService's existing
fail-fast handling of Llm errors (docs/adr/0009-generation-strategy.md).
Retries on transient failures are left to the SDK's own built-in retry
behavior; no custom retry logic is implemented.
"""


class OpenAiLlm:
    def __init__(self, api_key: str, model: str, timeout: float) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, timeout=timeout)
        self._model = model

    def generate(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""
