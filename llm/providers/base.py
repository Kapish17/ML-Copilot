"""The provider contract.

Everything above this interface — prompt building, grounding, the answer
service — depends on it and on nothing else. Swapping OpenAI for a local model
behind an OpenAI-compatible endpoint, or for a different vendor entirely, is a
change to which provider is constructed.

A provider does one thing: take a :class:`~llm.messages.GenerationRequest` and
return a :class:`~llm.messages.GenerationResult`. It does not know what
retrieval is, what a citation is, or that grounding exists. It does not import
pandas, scikit-learn or anything from ``ml/`` or ``rag/``.

Three obligations every implementation carries:

**Laziness.** Importing the module and constructing the provider must not
build an SDK client, read a credential or touch the network. The first
generation call does all of that. This is what lets the package import, and
the whole test suite run, with no key configured.

**Typed failures.** Every failure leaves as an :class:`~llm.errors.LLMError`
subclass. A caller must never have to catch a vendor's exception class, and a
vendor's exception must never reach a user.

**No credential leakage.** The key is read from the environment when needed
and used; it is never stored on the provider, put in a message, echoed in an
error or written to a log.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from llm.messages import GenerationRequest, GenerationResult


@runtime_checkable
class LLMProvider(Protocol):
    """Turns a request into generated text."""

    @property
    def name(self) -> str:
        """Stable identifier of this provider, e.g. ``"openai"``.

        Reported on every answer, so a reader can tell which service produced
        a claim.
        """
        ...  # pragma: no cover - protocol

    @property
    def is_ready(self) -> bool:
        """Whether a generation call could be attempted right now.

        False when a credential is missing. Checking this never contacts the
        provider and never reveals the credential — it answers "is it
        configured", not "is it valid", which only an actual call can tell.
        """
        ...  # pragma: no cover - protocol

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate a completion.

        Args:
            request: The messages and the settings for this call.

        Returns:
            GenerationResult: The generated text and its metadata.

        Raises:
            LLMConfigurationError: If no credential is configured, or the SDK
                is not installed. Nothing was attempted.
            LLMAuthenticationError: If the provider rejected the credential.
            LLMTimeoutError: If the provider did not answer in time.
            LLMRateLimitError: If the provider is throttling.
            LLMContextTooLargeError: If the request exceeded the context
                window.
            LLMUnavailableError: If the provider could not be reached.
            LLMResponseError: If the provider answered with nothing usable.
        """
        ...  # pragma: no cover - protocol


__all__ = ["LLMProvider"]
