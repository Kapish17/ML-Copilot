"""Language-model providers, and how one is chosen.

``base``             the :class:`LLMProvider` contract
``openai_provider``  the real one: any OpenAI-compatible chat API
``fake``             a deterministic, scriptable provider for tests

Providers are resolved by name from an :class:`~llm.config.LLMConfig`, so the
choice lives in configuration and nothing downstream names a vendor. Adding a
provider means writing a class that satisfies the protocol and registering it
here.

Construction is free of side effects: no SDK is imported, no credential is
read and no network call is made until a generation is actually requested.
"""

from __future__ import annotations

from llm.config import PROVIDER_FAKE, PROVIDER_OPENAI, AVAILABLE_PROVIDERS, LLMConfig
from llm.errors import LLMConfigurationError
from llm.providers.base import LLMProvider
from llm.providers.fake import FakeLLMProvider


def build_llm_provider(config: LLMConfig) -> LLMProvider:
    """Construct the provider a configuration asks for.

    Nothing is loaded, read or contacted here — every provider defers that to
    its first generation call, which is what lets this package be imported and
    tested with no credential present.

    Args:
        config: Names the provider, the model and the endpoint.

    Returns:
        LLMProvider: A ready-to-use provider.

    Raises:
        LLMConfigurationError: If the provider name is unknown.
    """
    name = config.provider.strip().lower()

    if name == PROVIDER_OPENAI:
        # Imported here so that constructing a fake provider, or importing
        # this package at all, never pulls in the SDK.
        from llm.providers.openai_provider import OpenAIProvider

        return OpenAIProvider(config)

    if name == PROVIDER_FAKE:
        return FakeLLMProvider(model=config.model)

    raise LLMConfigurationError(
        f"Unknown LLM provider '{config.provider}'. Available: "
        + ", ".join(AVAILABLE_PROVIDERS)
        + ".",
        details={"provider": config.provider, "available": list(AVAILABLE_PROVIDERS)},
    )


__all__ = [
    "AVAILABLE_PROVIDERS",
    "FakeLLMProvider",
    "LLMProvider",
    "build_llm_provider",
]
