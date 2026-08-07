"""
Verifies the LangGraph *routing* itself: which nodes ran and in what order,
for each classification path, plus the retry and safe-failure edges. None
of these assertions depend on the exact wording of a generated answer -
they check `node_trace`, `classification`, `requires_human`, and
`verification_result`, which satisfies the assignment's requirement for at
least one wording-independent routing test.
"""
import pytest

from graph.workflow import build_graph, initial_state
from retrieval.retriever import Retriever
from tests.fakes import AlwaysUngroundedGenerationModel, FailsOnceThenGroundedGenerationModel


@pytest.fixture(scope="module")
def retriever():
    return Retriever.build(offline=True)


def test_out_of_scope_never_reaches_retrieval(retriever):
    app, _ = build_graph(retriever, offline=True)
    result = app.invoke(initial_state("Please issue a refund for my subscription."))
    assert result["final_response"]["classification"] == "out_of_scope"
    assert "retrieval" not in result["node_trace"]
    assert "generation" not in result["node_trace"]


def test_vague_question_routes_to_clarification_without_generation(retriever):
    app, _ = build_graph(retriever, offline=True)
    result = app.invoke(initial_state("Our data sync is not working. Fix it please."))
    assert result["final_response"]["classification"] == "requires_clarification"
    assert result["final_response"]["clarification_question"] is not None
    assert "generation" not in result["node_trace"]


def test_escalation_question_sets_requires_human(retriever):
    app, _ = build_graph(retriever, offline=True)
    q = "Two export runs in a row failed with render_failed after we already checked everything."
    result = app.invoke(initial_state(q))
    assert result["final_response"]["classification"] == "requires_escalation"
    assert result["final_response"]["requires_human"] is True


def test_retry_edge_is_actually_taken_and_then_succeeds(retriever):
    fake_model = FailsOnceThenGroundedGenerationModel()
    app, _ = build_graph(retriever, offline=True, generation_model=fake_model)
    result = app.invoke(initial_state("Can a Viewer create an API credential?"))

    assert fake_model.calls == 2, "generation should be called once, fail, then retried once"
    assert result["node_trace"].count("generation") == 2
    assert "prepare_retry" in result["node_trace"]
    assert result["final_response"]["classification"] == "answerable"


def test_verification_failure_exhausts_retry_and_returns_safe_failure(retriever):
    fake_model = AlwaysUngroundedGenerationModel()
    app, _ = build_graph(retriever, offline=True, generation_model=fake_model)
    result = app.invoke(initial_state("Can a Viewer create an API credential?"))

    # loop protection: generation is called exactly retries+1 times, never more
    assert result["node_trace"].count("generation") == 2
    assert result["final_response"]["classification"] == "safe_failure"
    assert result["final_response"]["requires_human"] is True
    assert result["verification_result"]["passed"] is False


def test_low_retrieval_confidence_downgrades_to_clarification(retriever, monkeypatch):
    import config

    monkeypatch.setattr(config.RETRIEVAL_CONFIG, "low_confidence_threshold", 1.1)  # force downgrade
    app, _ = build_graph(retriever, offline=True)
    result = app.invoke(initial_state("What should we check about exports?"))
    assert result["final_response"]["classification"] == "requires_clarification"
    assert "confidence_gate" in result["node_trace"]
    assert "generation" not in result["node_trace"]


def test_followup_question_is_rewritten_using_previous_question(retriever):
    app, _ = build_graph(retriever, offline=True)
    state = initial_state(
        "What about Admins?",
        previous_question="Can a read-only Viewer create an API credential?",
    )
    result = app.invoke(state)

    assert "query_rewrite" in result["node_trace"]
    assert result.get("rewritten_query") is not None
    assert "Admins" in result["rewritten_query"] or "admins" in result["rewritten_query"].lower()
    assert "Viewer" in result["rewritten_query"]


def test_self_contained_question_is_not_rewritten_even_with_history(retriever):
    app, _ = build_graph(retriever, offline=True)
    state = initial_state(
        "Our exports stopped after the workspace timezone changed and we need to know what to check.",
        previous_question="Can a read-only Viewer create an API credential?",
    )
    result = app.invoke(state)
    assert result.get("rewritten_query") is None
