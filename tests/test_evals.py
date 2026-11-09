"""Eval harness tests: scores measure diagnosis correctness, and a
deliberately bad investigator scores badly."""

from network_intelligence.evals import load_golden_cases, run_evals


def test_golden_cases_load():
    cases = load_golden_cases()
    assert len(cases) == 4
    assert {c["expected_root_cause"] for c in cases} == {
        "kv_cache_exhaustion",
        "faulty_optic",
        "batching_too_small",
        "no-fault",
    }


def test_demo_agent_scores_perfectly(agent):
    result = run_evals(agent.investigate)
    assert result.root_cause_accuracy == 1.0
    assert result.false_alarm_rate == 0.0
    assert result.mean_evidence_score == 1.0
    assert len(result.cases) == 4


def test_always_no_fault_investigator_scores_badly():
    def investigate(_incident):
        return {"root_cause": "no-fault", "evidence": ["decision_tree -> no-fault"]}

    result = run_evals(investigate)
    assert result.root_cause_accuracy == 0.25  # only the healthy case
    assert result.false_alarm_rate == 0.0  # it never raises a false alarm
    assert result.mean_evidence_score < 0.5  # it cites no real evidence


def test_eval_result_render(agent):
    result = run_evals(agent.investigate)
    text = result.render_text()
    assert "root-cause accuracy: 100%" in text
    assert "false-alarm rate:    0%" in text
