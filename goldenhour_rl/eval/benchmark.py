import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
from goldenhour_rl.sandbox.verifier import verify_dispatch_plan


def greedy_nearest_heuristic(state: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Simulates a standard naive greedy dispatcher:
    Routes each patient in order to the nearest hospital via the first available ambulance,
    ignoring global ventilator and ICU capacity constraints (causes frequent bottlenecks).
    """
    patients = state.get("patients", [])
    ambulances = state.get("ambulances", [])
    hospitals = list(state.get("hospitals", {}).keys())

    plan = []
    for idx, pat in enumerate(patients):
        p_id = pat["patient_id"]
        amb = ambulances[idx % len(ambulances)]
        a_id = amb["ambulance_id"]

        # Naively pick the hospital with the lowest ETA
        best_hosp = min(hospitals, key=lambda h: amb["eta_matrix_mins"][p_id].get(h, 999))
        plan.append({
            "patient_id": p_id,
            "ambulance_id": a_id,
            "hospital_id": best_hosp,
        })
    return plan


def evaluate_dataset_on_strategy(dataset_path: str, strategy_name: str) -> Dict[str, Any]:
    """Evaluates a dispatch strategy over a full crisis dataset."""
    samples = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    total_scenarios = len(samples)
    zero_violation_count = 0
    total_violations = 0
    total_sla_margins = 0.0
    total_patients = 0
    total_breaches = 0
    total_icu_overflows = 0
    total_vent_overflows = 0

    for sample in samples:
        state = json.loads(sample["state_json"])

        if strategy_name == "reference":
            plan = json.loads(sample.get("reference_solution", "[]"))
        elif strategy_name == "greedy_nearest":
            plan = greedy_nearest_heuristic(state)
        elif strategy_name == "random":
            plan = []
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        ver_res = verify_dispatch_plan(plan, sample["state_json"])
        if ver_res["valid"]:
            zero_violation_count += 1
            total_sla_margins += ver_res["sla_score"]

        total_violations += len(ver_res["violations"])
        stats = ver_res.get("stats", {})
        total_patients += stats.get("total_patients", 0)
        total_breaches += stats.get("breached_count", 0)
        total_icu_overflows += stats.get("icu_overflow_count", 0)
        total_vent_overflows += stats.get("vent_overflow_count", 0)

    pass_rate = (zero_violation_count / total_scenarios) * 100 if total_scenarios > 0 else 0.0
    avg_margin = (total_sla_margins / zero_violation_count) if zero_violation_count > 0 else 0.0
    patient_survival_rate = ((total_patients - total_breaches) / total_patients * 100) if total_patients > 0 else 0.0

    return {
        "strategy": strategy_name,
        "scenarios": total_scenarios,
        "zero_violation_rate": round(pass_rate, 2),
        "patient_survival_sla_rate": round(patient_survival_rate, 2),
        "total_violations": total_violations,
        "icu_overflows": total_icu_overflows,
        "vent_overflows": total_vent_overflows,
        "sla_breaches": total_breaches,
        "avg_sla_margin": round(avg_margin, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="GoldenHour-RL Dispatch Benchmark")
    parser.add_argument("--eval_data", type=str, default="data/crisis_eval.jsonl")
    parser.add_argument("--dry_run", action="store_true", help="Run benchmark across heuristic baselines")
    args = parser.parse_args()

    eval_path = Path(args.eval_data)
    if not eval_path.exists():
        print(f"[GoldenHour-RL Benchmark] Eval dataset not found at {eval_path}. Generating 50 eval samples...")
        from goldenhour_rl.data.crisis_generator import generate_scenario
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        with eval_path.open("w", encoding="utf-8") as f:
            for i in range(50):
                f.write(json.dumps(generate_scenario(9000 + i)) + "\n")

    print("\n" + "=" * 80)
    print("           GOLDENHOUR-RL CRISIS DISPATCH BENCHMARK TELEMETRY")
    print("=" * 80)

    strategies = ["greedy_nearest", "reference"]
    results = []

    for strat in strategies:
        res = evaluate_dataset_on_strategy(str(eval_path), strat)
        results.append(res)

    header = f"{'Strategy':<20} | {'Zero-Viol %':<12} | {'Survival %':<12} | {'ICU Overflows':<14} | {'Vent Overflows':<14} | {'Violations'}"
    print(header)
    print("-" * len(header))

    for r in results:
        line = (
            f"{r['strategy']:<20} | "
            f"{r['zero_violation_rate']:<10.1f}% | "
            f"{r['patient_survival_sla_rate']:<10.1f}% | "
            f"{r['icu_overflows']:<14} | "
            f"{r['vent_overflows']:<14} | "
            f"{r['total_violations']}"
        )
        print(line)

    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
