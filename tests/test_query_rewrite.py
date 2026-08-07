from nodes.query_rewrite import effective_query, is_probable_followup


def test_short_question_is_treated_as_followup():
    assert is_probable_followup("What about admins?") is True
    assert is_probable_followup("And Viewers too?") is True


def test_long_self_contained_question_is_not_a_followup():
    q = (
        "Our exports stopped after the workspace timezone changed last week "
        "and we need to know exactly what settings to check before contacting support."
    )
    assert is_probable_followup(q) is False


def test_effective_query_falls_back_to_question_when_no_rewrite():
    state = {"question": "Can a Viewer create an API credential?"}
    assert effective_query(state) == "Can a Viewer create an API credential?"


def test_effective_query_prefers_rewritten_query():
    state = {"question": "What about admins?", "rewritten_query": "Can a Viewer create a credential? Follow-up: What about admins?"}
    assert effective_query(state) == state["rewritten_query"]
