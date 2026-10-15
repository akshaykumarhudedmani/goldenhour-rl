import json
from goldenhour_rl.data.crisis_generator import generate_scenario
from goldenhour_rl.sandbox.executor import execute_dispatch_code, validate_ast
from goldenhour_rl.sandbox.verifier import verify_dispatch_plan
from goldenhour_rl.training.rewards import format_reward_func, verifier_reward_func


def test_infinite_loop_killed():
    """Verify that execution sandbox terminates infinite loops within timeout without hanging."""
    malicious_code = "def get_dispatch_plan():\n    while True:\n        pass\n"
    res = execute_dispatch_code(malicious_code, timeout_sec=0.5)
    assert res["ok"] is False
    assert res["error"] == "ExecutionTimeout"


def test_ast_security_catches_imports_and_banned_functions():
    """Verify AST security checks catch dangerous imports and unsafe builtins."""
    import_code = "import os\ndef get_dispatch_plan():\n    return []\n"
    assert "SecurityViolation" in validate_ast(import_code)

    eval_code = "def get_dispatch_plan():\n    eval('2+2')\n    return []\n"
    assert "SecurityViolation" in validate_ast(eval_code)


def test_reference_solution_is_valid():
    """Verify that synthetic scenarios always include a physically valid reference solution."""
    sample = generate_scenario(seed=777)
    ref_plan = json.loads(sample["reference_solution"])
    ver_res = verify_dispatch_plan(ref_plan, sample["state_json"])
    assert ver_res["valid"] is True
    assert len(ver_res["violations"]) == 0
    assert ver_res["sla_score"] >= 0.0


def test_hallucinated_hospital_caught():
    sample = generate_scenario(seed=101)
    state = json.loads(sample["state_json"])
    bad_plan = [
        {"patient_id": p["patient_id"], "ambulance_id": "AMB_1", "hospital_id": "HOSP_GHOST"}
        for p in state["patients"]
    ]
    res = verify_dispatch_plan(bad_plan, sample["state_json"])
    assert res["valid"] is False
    assert any("Hallucination" in v for v in res["violations"])


def test_double_booked_ambulance_caught():
    sample = generate_scenario(seed=102)
    state = json.loads(sample["state_json"])
    # Assign the exact same ambulance to all patients
    amb_id = state["ambulances"][0]["ambulance_id"]
    hosp_id = list(state["hospitals"].keys())[0]
    bad_plan = [
        {"patient_id": p["patient_id"], "ambulance_id": amb_id, "hospital_id": hosp_id}
        for p in state["patients"]
    ]
    res = verify_dispatch_plan(bad_plan, sample["state_json"])
    assert res["valid"] is False
    assert any("DoubleBooking" in v for v in res["violations"])


def test_reward_pipeline_end_to_end():
    sample = generate_scenario(seed=202)
    valid_plan_json = sample["reference_solution"]

    mock_valid_completion = (
        "<think>Analyzing patient survival windows and ventilator availability across hospitals.</think>\n"
        f"```python\n"
        f"def get_dispatch_plan():\n"
        f"    return {valid_plan_json}\n"
        f"```"
    )

    # Format reward check
    fmt_scores = format_reward_func([sample["prompt"]], [mock_valid_completion])
    assert fmt_scores[0] > 0.0

    # Verifier reward check (should give high positive reward)
    ver_scores = verifier_reward_func(
        [sample["prompt"]],
        [mock_valid_completion],
        state_json=[sample["state_json"]]
    )
    assert ver_scores[0] >= 2.0


def test_malformed_completion_penalized():
    sample = generate_scenario(seed=303)
    mock_bad_completion = "I would assign the first ambulance to patient 1."

    fmt_scores = format_reward_func([sample["prompt"]], [mock_bad_completion])
    assert fmt_scores[0] < 0.0

    ver_scores = verifier_reward_func(
        [sample["prompt"]],
        [mock_bad_completion],
        state_json=[sample["state_json"]]
    )
    assert ver_scores[0] < 0.0
