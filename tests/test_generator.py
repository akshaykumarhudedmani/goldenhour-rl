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
        assert len(state["hospitals"]) >= 3

        # Verify reference solution satisfies all invariants
        ver_res = verify_dispatch_plan(ref_plan, sample["state_json"])
        assert ver_res["valid"] is True, f"Seed {seed} generated an invalid reference plan: {ver_res['violations']}"
        assert len(ver_res["violations"]) == 0
