"""ML Copilot language-model layer.

Turns retrieved evidence into a grounded answer with citations.

::

    question → retrieve → evidence → prompt → model → validate citations
                                                    → grounded answer

**The model is not the source of truth; retrieved evidence is.** The model's
knowledge is used to explain what an F1 score means, never to supply what this
project scored. Every project-specific claim must come from a retrieved
passage, every citation is checked against the passages actually supplied, and
an answer citing a source that was not retrieved is rejected rather than
quietly cleaned up.

The layer is independent: it imports neither FastAPI nor the backend, and it
does not import pandas, scikit-learn or anything from ``ml/``. It consumes the
structured :class:`~rag.retrieval.RetrievalResult` objects that ``rag/``
produces, and ``rag/`` knows nothing about it.

Everything is offline-safe. Importing this package builds no client, reads no
credential and contacts nothing; the whole test suite runs with no API key,
against a deterministic fake provider. **No agent, LangGraph, autonomous tool
calling or multi-agent system is implemented.**

Typical use::

    from llm import LLMConfig, RAGAnswerService, build_llm_provider
    from rag import RagConfig, RetrievalService

    config = LLMConfig()
    service = RAGAnswerService(
        config,
        retriever=RetrievalService(RagConfig()),
        provider=build_llm_provider(config),
    )

    answer = service.answer("Which model was selected in experiment exp_123?")
    print(answer.status.value, answer.answer)
    for citation in answer.citations:
        print(" ", citation.citation_id, citation.source_title)
"""

from llm.answers import Answer, AnswerMetadata, AnswerStatus, Citation
from llm.config import (
    AVAILABLE_PROVIDERS,
    PROVIDER_FAKE,
    PROVIDER_OPENAI,
    LLMConfig,
    config_from_env,
)
from llm.context import EvidenceContext, build_context
from llm.errors import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMContextTooLargeError,
    LLMDependencyError,
    LLMError,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from llm.grounding import GroundingReport, extract_citations, validate_citations
from llm.messages import GenerationRequest, GenerationResult, Message, Role
from llm.prompts import (
    INSUFFICIENT_EVIDENCE_ANSWER,
    INSUFFICIENT_EVIDENCE_MARKER,
    SYSTEM_PROMPT,
    build_system_prompt,
    build_user_prompt,
)
from llm.providers import FakeLLMProvider, LLMProvider, build_llm_provider
from llm.service import RAGAnswerService, Retriever

__all__ = [
    "AVAILABLE_PROVIDERS",
    "INSUFFICIENT_EVIDENCE_ANSWER",
    "INSUFFICIENT_EVIDENCE_MARKER",
    "PROVIDER_FAKE",
    "PROVIDER_OPENAI",
    "SYSTEM_PROMPT",
    "Answer",
    "AnswerMetadata",
    "AnswerStatus",
    "Citation",
    "EvidenceContext",
    "FakeLLMProvider",
    "GenerationRequest",
    "GenerationResult",
    "GroundingReport",
    "LLMAuthenticationError",
    "LLMConfig",
    "LLMConfigurationError",
    "LLMContextTooLargeError",
    "LLMDependencyError",
    "LLMError",
    "LLMProvider",
    "LLMProviderError",
    "LLMRateLimitError",
    "LLMResponseError",
    "LLMTimeoutError",
    "LLMUnavailableError",
    "Message",
    "RAGAnswerService",
    "Retriever",
    "Role",
    "build_context",
    "build_llm_provider",
    "build_system_prompt",
    "build_user_prompt",
    "config_from_env",
    "extract_citations",
    "validate_citations",
]
