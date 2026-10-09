"""Tests for the evaluation suite (Part 2, LLMOps)."""
from app.eval.evaluator import (
    evaluate_classification,
    run_full_evaluation,
    score_response_qa,
)
from app.llm.client import LLMClient


def test_classification_accuracy_meets_target():
    llm = LLMClient(api_key="")
    report = evaluate_classification(llm)
    assert report["accuracy"] >= 0.8, report
    assert report["total"] == 30


def test_full_evaluation_report_shape():
    report = run_full_evaluation()
    assert report["classification"]["accuracy"] >= 0.8
    assert report["routing"]["success_rate"] >= 0.8
    assert set(report["response_quality"]) == {
        "positive_feedback", "negative_feedback", "query"
    }
    for kind, metrics in report["response_quality"].items():
        assert metrics["n"] == 10
        for key, value in metrics.items():
            if key == "n":
                continue
            assert 0.0 <= value <= 1.0, (kind, key, value)


def test_qa_scoring_bounds():
    s = score_response_qa("negative_feedback",
                          "We're so sorry, we understand your frustration. "
                          "Ticket 123456 has been generated, and our team will follow up shortly.",
                          ticket_id="123456")
    assert s["empathy"] > 0
    assert s["ticket_reference"] == 1.0
    assert 0.0 <= s["overall"] <= 1.0
