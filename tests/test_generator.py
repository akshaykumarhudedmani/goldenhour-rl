import json
from goldenhour_rl.data.crisis_generator import generate_scenario
from goldenhour_rl.sandbox.verifier import verify_dispatch_plan


def test_scenario_feasibility_across_random_seeds():
    """
    Stress-test the generator across 25 different seeds.
    Every scenario must mathematically guarantee at least one 100% valid dispatch plan.
    """
    for seed in range(100, 125):
        sample = generate_scenario(seed)
        state = json.loads(sample["state_json"])
        ref_plan = json.loads(sample["reference_solution"])

        # Check telemetry integrity
        assert len(state["patients"]) >= 3
        assert len(state["ambulances"]) >= len(state["patients"])
        assert len(state["hospitals"]) >= 2

        # Verify reference solution satisfies all invariants
        ver_res = verify_dispatch_plan(ref_plan, sample["state_json"])
        assert ver_res["valid"] is True, f"Seed {seed} generated an invalid reference plan: {ver_res['violations']}"
        assert len(ver_res["violations"]) == 0


def test_zero_leakage_and_trap_routes():
    """Verify that non-reference routes also contain short ETAs so no statistical cue leakage exists."""
    trap_routes_found = 0
    for seed in range(50):
        sample = generate_scenario(seed=500 + seed)
        state = json.loads(sample["state_json"])
        ref_plan = json.loads(sample["reference_solution"])
        ref_pairs = {(p["patient_id"], p["ambulance_id"]) for p in ref_plan}

        for amb in state["ambulances"]:
            a_id = amb["ambulance_id"]
            for p_id, routes in amb["eta_matrix_mins"].items():
                if (p_id, a_id) not in ref_pairs:
                    # Check if non-reference route has low ETA (< 20 mins)
                    if any(eta < 20 for eta in routes.values()):
                        trap_routes_found += 1
                        break

    # Trap routes must be actively present across the dataset
    assert trap_routes_found > 25, "Trap routes are missing, risk of statistical cue leakage!"


def test_verified_sft_sample_generation():
    """Verify that SFT dataset generator produces 100% verified samples with valid ChatML messages."""
    from goldenhour_rl.data.sft_generator import generate_verified_sft_sample
    from goldenhour_rl.sandbox.executor import execute_dispatch_code
    from goldenhour_rl.eval.benchmark import extract_python_code

    for seed in range(5):
        sft_sample = generate_verified_sft_sample(seed=700 + seed)
        assert "messages" in sft_sample
        assert len(sft_sample["messages"]) == 2
        assert sft_sample["messages"][0]["role"] == "user"
        assert sft_sample["messages"][1]["role"] == "assistant"

        # Check reasoning trace and code block in completion
        comp = sft_sample["completion"]
        assert "<think>" in comp and "</think>" in comp
        assert "```python" in comp

        # Extract code and verify it executes with 0 violations
        code = extract_python_code(comp)
        assert code is not None
        exec_res = execute_dispatch_code(code, timeout_sec=2.0, state_str=sft_sample["state_json"])
        assert exec_res["ok"] is True
        ver_res = verify_dispatch_plan(exec_res["plan"], sft_sample["state_json"])
        assert ver_res["valid"] is True
        assert len(ver_res["violations"]) == 0

