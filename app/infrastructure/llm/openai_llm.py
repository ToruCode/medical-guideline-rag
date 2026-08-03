"""OpenAI Chat Completions-based implementation of the Llm port.

Importing openai is deferred to __init__ (not done at module import
time), so merely referencing the OpenAiLlm class - as
app/api/dependencies.py always does, regardless of which llm_provider
is configured - never pays the cost of importing the SDK. See
docs/adr/0011-real-embedding-and-llm-adapters.md.

Any exception raised by the OpenAI client is translated to
LlmGenerationError (never the raw SDK exception) so callers depend only
on app.domain.exceptions.llm, not on the openai package's exception
hierarchy; see docs/adr/0022-context-length-control-and-llm-error-handling.md.
The translated exception's message never includes the API key (it is
never part of any OpenAI SDK exception's message) and only names the
underlying exception's type, not its full text, as a defensive measure
against a future SDK version embedding request details in its message.
GenerateAnswerService still propagates whatever the Llm raises
unchanged, matching its existing fail-fast handling
(docs/adr/0009-generation-strategy.md). Retries on transient failures
are left to the SDK's own built-in retry behavior; no custom retry
logic is implemented.
"""

from app.domain.exceptions.llm import LlmGenerationError


class OpenAiLlm:
    def __init__(self, api_key: str, model: str, timeout: float) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, timeout=timeout)
        self._model = model

    def generate(self, prompt: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:
            raise LlmGenerationError(f"OpenAI request failed: {type(exc).__name__}") from exc
        return response.choices[0].message.content or ""
