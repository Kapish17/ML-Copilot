"""What is sent to a model, and what comes back.

These types are the vocabulary the provider interface speaks. They are
deliberately plain — a role, some text, a few settings — so that a provider
implementation is a translation into one SDK's shapes and nothing more, and so
that swapping providers cannot change what the rest of the layer sees.

Nothing here knows about retrieval, citations or grounding. A
:class:`GenerationRequest` is a system prompt, a user prompt and some limits;
building the user prompt from evidence happens in :mod:`llm.prompts`, and
checking what comes back happens in :mod:`llm.grounding`. Keeping those apart
is what lets the grounding rules be tested without a model and the provider be
tested without any grounding.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Role(str, Enum):
    """Who a message is from."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True)
class Message:
    """One turn of a conversation."""

    role: Role
    content: str

    def as_dict(self) -> dict[str, str]:
        """Render the message in the shape chat APIs expect."""
        return {"role": self.role.value, "content": self.content}


@dataclass(frozen=True)
class GenerationRequest:
    """One request to a model.

    The settings are carried on the request rather than read from a
    configuration inside the provider, so a caller can vary temperature or
    length per call and a test can assert exactly what was asked for.
    """

    messages: tuple[Message, ...]
    model: str
    temperature: float = 0.0
    max_output_tokens: int = 900
    timeout_seconds: float = 30.0
    #: Free-form provider hints. Kept out of the typed fields because what one
    #: provider supports another does not.
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def system_prompt(self) -> str:
        """The system message's content, or an empty string."""
        for message in self.messages:
            if message.role is Role.SYSTEM:
                return message.content
        return ""

    @property
    def user_prompt(self) -> str:
        """The last user message's content, or an empty string."""
        for message in reversed(self.messages):
            if message.role is Role.USER:
                return message.content
        return ""

    @property
    def character_count(self) -> int:
        """Total characters across every message."""
        return sum(len(message.content) for message in self.messages)

    def as_payload(self) -> list[dict[str, str]]:
        """Render the messages for a chat-completions style API."""
        return [message.as_dict() for message in self.messages]


@dataclass(frozen=True)
class GenerationResult:
    """What a model returned.

    Holds the text and a little metadata, and deliberately no provider object,
    no raw response and no request echo — those carry headers, keys and
    internals that have no business travelling further into the system.
    """

    text: str
    model: str
    provider: str
    #: Why the model stopped, when the provider says: ``"stop"``, ``"length"``,
    #: a filter. ``None`` when it does not.
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_seconds: float | None = None

    @property
    def is_truncated(self) -> bool:
        """True when the model stopped because it hit the output limit."""
        return self.finish_reason == "length"

    @property
    def total_tokens(self) -> int | None:
        """Prompt plus completion tokens, when the provider reported both."""
        if self.prompt_tokens is None or self.completion_tokens is None:
            return None
        return self.prompt_tokens + self.completion_tokens

    def as_dict(self) -> dict[str, Any]:
        """Render the result as plain JSON-safe values."""
        return {
            "text": self.text,
            "model": self.model,
            "provider": self.provider,
            "finish_reason": self.finish_reason,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency_seconds": self.latency_seconds,
        }


def build_messages(system_prompt: str, user_prompt: str) -> tuple[Message, ...]:
    """Build the two-message conversation this layer always sends.

    One system message carrying the rules and one user message carrying the
    question and its evidence. There is no conversation history: each answer
    is grounded in the evidence retrieved for that question, and carrying
    earlier turns would let an ungrounded claim from one answer become the
    premise of the next.
    """
    messages = []
    if system_prompt.strip():
        messages.append(Message(role=Role.SYSTEM, content=system_prompt))
    messages.append(Message(role=Role.USER, content=user_prompt))
    return tuple(messages)


def redact(text: str, secrets: Sequence[str]) -> str:
    """Replace any occurrence of a secret with a placeholder.

    A defence in depth, not the primary one: keys are never put into prompts
    or errors in the first place. This exists so that if a credential ever
    reaches a string that is about to be logged or returned, it does not
    survive the trip.

    Args:
        text: The text to clean.
        secrets: Values that must not appear. Empty and very short values are
            ignored, since redacting them would mangle ordinary text.

    Returns:
        str: The text with every secret replaced by ``"[redacted]"``.
    """
    cleaned = text
    for secret in secrets:
        if secret and len(secret) >= 8:
            cleaned = cleaned.replace(secret, "[redacted]")
    return cleaned


__all__ = [
    "GenerationRequest",
    "GenerationResult",
    "Message",
    "Role",
    "build_messages",
    "redact",
]
