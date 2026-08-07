from verification.checks import check_evidence_support, check_schema, has_citations
from models.embedding_model import HashingEmbedder


def _doc(source_id: str, passage: str):
    return {
        "source_id": source_id,
        "title": "t",
        "passage": passage,
        "score": 0.9,
        "vector_score": 0.9,
        "keyword_score": 0.9,
        "is_superseded": False,
    }


def test_grounded_answer_has_high_overlap():
    docs = [_doc("KB-005", "An Owner or Admin can create a credential from Settings, Developer, API credentials.")]
    answer = "An Owner or Admin can create a credential from Settings [KB-005]."
    result = check_evidence_support(answer, docs)
    assert result.evidence_overlap >= 0.5


def test_hallucinated_answer_has_low_overlap():
    docs = [_doc("KB-005", "An Owner or Admin can create a credential from Settings, Developer, API credentials.")]
    answer = "OrbitDesk will mail you a physical printed backup copy of your dashboard every month."
    result = check_evidence_support(answer, docs)
    assert result.evidence_overlap < 0.3
    assert len(result.unsupported_sentences) == 1


def test_has_citations_detects_bracket_source_id():
    assert has_citations("Resave the schedule [KB-003].") is True
    assert has_citations("Resave the schedule.") is False


def test_schema_rejects_answerable_without_sources():
    payload = {
        "classification": "answerable",
        "answer": "Do the thing.",
        "sources": [],
        "confidence": 0.9,
        "requires_human": False,
        "reason": "test",
    }
    valid, errors = check_schema(payload)
    assert valid is False
    assert errors


def test_schema_accepts_valid_payload():
    payload = {
        "classification": "answerable",
        "answer": "Do the thing.",
        "sources": [{"source_id": "KB-001", "passage": "excerpt"}],
        "confidence": 0.9,
        "requires_human": False,
        "reason": "test",
    }
    valid, errors = check_schema(payload)
    assert valid is True
    assert errors == []


def test_semantic_guard_accepts_grounded_answer():
    embedder = HashingEmbedder()
    docs = [_doc("KB-005", "An Owner or Admin can create a credential from Settings, Developer, API credentials.")]
    answer = "An Owner or Admin is able to create a credential from the Settings menu [KB-005]."
    result = check_evidence_support(answer, docs, min_overlap=0.9, embedder=embedder, semantic_threshold=0.3)
    assert result.evidence_overlap == 1.0
    assert result.unsupported_sentences == []


def test_semantic_guard_rejects_unrelated_claim():
    embedder = HashingEmbedder()
    docs = [_doc("KB-005", "An Owner or Admin can create a credential from Settings, Developer, API credentials.")]
    answer = "OrbitDesk will mail you a physical printed backup copy of your dashboard every month."
    result = check_evidence_support(answer, docs, min_overlap=0.9, embedder=embedder, semantic_threshold=0.9)
    assert result.evidence_overlap < 1.0
    assert len(result.unsupported_sentences) == 1


def test_semantic_guard_handles_empty_retrieved_docs():
    embedder = HashingEmbedder()
    result = check_evidence_support("Some answer sentence here for testing.", [], embedder=embedder)
    assert result.evidence_overlap == 0.0


def test_paraphrase_can_pass_via_semantic_layer_even_with_low_lexical_overlap():
    """A correct paraphrase with low word overlap should not be penalised
    when a semantic embedder is available, even if the lexical layer alone
    would flag it."""
    embedder = HashingEmbedder()
    docs = [_doc("KB-002", "An Analyst cannot invite members, change workspace settings, reveal stored secrets or create API credentials.")]
    # Same meaning, almost entirely different words from the passage.
    answer = "Analysts are not permitted to reveal stored secrets or create API credentials [KB-002]."
    lexical_only = check_evidence_support(answer, docs, min_overlap=0.9)  # very strict lexical bar
    with_semantic = check_evidence_support(answer, docs, min_overlap=0.9, embedder=embedder, semantic_threshold=0.2)
    assert with_semantic.evidence_overlap >= lexical_only.evidence_overlap
