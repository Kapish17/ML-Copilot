"""An optional smoke test of POST /api/v1/ask against a real provider.

**Not part of the normal test suite.** It is skipped unless a credential is
configured *and* the test is explicitly opted into, so an ordinary run stays
offline, free and deterministic — a developer who happens to have a key in
their environment should not start spending it by running ``pytest``.

To run it::

    export LLM_API_KEY=...
    export RUN_LLM_INTEGRATION=1
    pytest backend/tests/test_api_knowledge_real_provider.py -m external

``llm/tests/test_real_provider.py`` already covers the library path. This one
exists for the part only the endpoint can break: that a real provider reached
through dependency injection, over HTTP, still produces a response matching
the published schema — and that a real model's citations survive validation
rather than being rejected by an over-strict parser that a scripted fake
happens to satisfy.

The assertions check the *shape* of what came back, never its wording: a real
model is not deterministic, and a test that pinned its prose would fail for
the wrong reason. The key is never printed, asserted on, or included in any
output.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from llm.config import LLMConfig
from rag.config import RagConfig
from rag.indexing import RagIndexer
from rag.stores import LocalVectorStore

ASK_URL = "/api/v1/ask"

#: Set to "1" to opt in. A configured key alone is deliberately not enough.
OPT_IN_VARIABLE = "RUN_LLM_INTEGRATION"

#: A question this project's own documentation genuinely answers.
QUESTION = "How does this project prevent data leakage between train and test?"

pytestmark = pytest.mark.external


def _enabled() -> bool:
    """Whether the opt-in and a credential are both present."""
    if os.getenv(OPT_IN_VARIABLE, "").strip() != "1":
        return False
    return LLMConfig().has_api_key


requires_credentials = pytest.mark.skipif(
    not _enabled(),
    reason=(
        f"Set {OPT_IN_VARIABLE}=1 and configure an API key to run the real "
        "provider smoke test. The rest of the suite runs offline."
    ),
)


@requires_credentials
def test_the_ask_endpoint_answers_through_a_real_provider(tmp_path: Path) -> None:
    """The endpoint returns a schema-shaped answer, and no credential."""
    index_dir = tmp_path / "index"
    rag_config = RagConfig(index_dir=index_dir)
    RagIndexer(rag_config, store=LocalVectorStore(index_dir)).index_documentation()

    client = TestClient(create_app(rag_config=rag_config))
    response = client.post(ASK_URL, json={"question": QUESTION, "top_k": 4})

    assert response.status_code == 200, response.text
    payload = response.json()

    # The contract, not the prose.
    assert payload["question"] == QUESTION
    assert isinstance(payload["answer"], str) and payload["answer"].strip()
    assert payload["status"] in {
        "grounded",
        "insufficient_evidence",
        "grounding_failed",
    }
    assert payload["is_grounded"] is (payload["status"] == "grounded")
    assert payload["metadata"]["retrieved_count"] > 0

    # Every citation the model produced was one it was actually given.
    allowed = set(payload["allowed_citations"])
    assert set(payload["citation_ids"]) <= allowed
    assert not set(payload["rejected_citations"]) & allowed

    # A real provider is a new way for a secret to escape.
    key = os.environ.get("LLM_API_KEY", "")
    body = json.dumps(payload)
    assert key and key not in body
    assert "sk-" not in body
    for marker in ("Traceback", "openai.", "object at 0x", str(tmp_path)):
        assert marker not in body, marker
