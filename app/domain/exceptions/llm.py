"""Exceptions raised while generating text via an Llm."""


class LlmError(Exception):
    """Base class for Llm failures."""


class LlmGenerationError(LlmError):
    """Raised when a concrete Llm implementation fails to generate a
    completion (e.g. a network error or an API error from a real
    provider). Must never include an API key or other credential in its
    message.
    """
