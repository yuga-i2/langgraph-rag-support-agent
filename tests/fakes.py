"""
Test doubles for the generation model.

These let the graph-routing tests force a verification failure (and a
subsequent retry) deterministically, without depending on what a real or
even the mock extractive model happens to write. This is what satisfies
the assignment's "at least one automated test must verify graph routing
without depending on the exact wording produced by the model" requirement.
"""
from __future__ import annotations

from typing import List

from graph.state import RetrievedDoc
from models.generation_model import GenerationResult


class AlwaysUngroundedGenerationModel:
    """Never cites sources and states an unrelated claim -> always fails verification."""

    model_name = "fake-always-ungrounded"

    def generate(self, question: str, context_docs: List[RetrievedDoc]) -> GenerationResult:
        return GenerationResult(
            text="OrbitDesk will automatically email you a complimentary gift card for this issue.",
            model_name=self.model_name,
        )


class FailsOnceThenGroundedGenerationModel:
    """
    Fails verification on the first attempt (no citation), then produces a
    properly grounded, cited answer on the retry. Used to prove the retry
    edge is actually taken, not just configured.
    """

    model_name = "fake-fails-once"

    def __init__(self):
        self.calls = 0

    def generate(self, question: str, context_docs: List[RetrievedDoc]) -> GenerationResult:
        self.calls += 1
        if self.calls == 1:
            return GenerationResult(
                text="This will just work automatically, no need to check anything.",
                model_name=self.model_name,
            )
        if not context_docs:
            return GenerationResult(text="No supporting documentation was found.", model_name=self.model_name)
        doc = context_docs[0]
        sentence = doc["passage"].split(". ")[0].strip().rstrip(".")
        return GenerationResult(text=f"{sentence} [{doc['source_id']}].", model_name=self.model_name)
