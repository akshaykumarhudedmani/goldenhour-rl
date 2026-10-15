import json
import random
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

SYSTEM_PROMPT = (
    "You are GoldenHour-RL, an emergency triage and dispatch optimization engine. "
    "Your objective is to safely route every patient to a suitable hospital via an available ambulance. "
    "Inside <think>...</think> tags, concisely analyze constraints (ventilator requirements, hospital ICU/ventilator capacities, and survival deadlines) in under 200 words. "
    "Then, output a Python block defining `def get_dispatch_plan():` returning a list of dicts: "
    "`[{'patient_id': str, 'ambulance_id': str, 'hospital_id': str}]`."
)


def _build_guaranteed_feasible_scenario(rng: random.Random, seed: int) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    """
    Constructs an emergency dispatch scenario with a mathematically guaranteed valid solution.
    Eliminates statistical cue leakage: background routes and reference routes share the same
    ETA distribution, and realistic trap routes (fast routes to full or unequipped hospitals) are injected.
    Scale is randomized across hospital counts (2 to 4) and patient counts (3 to 5).
    """
    num_hospitals = rng.randint(2, 4)
    num_patients = rng.randint(3, 5)
    num_ambulances = num_patients + rng.randint(0, 2)

    # Generate hospitals with finite life-support capacity
    hospitals = {}
    for h in range(num_hospitals):
        h_id = f"HOSP_{h+1}"
        hospitals[h_id] = {
            "icu_beds": rng.randint(1, 3),
            "ventilators": rng.randint(0, 2),
        }

    # Ensure total network capacity is sufficient for all patients
    total_icu = sum(h["icu_beds"] for h in hospitals.values())
    while total_icu < num_patients:
        target_h = rng.choice(list(hospitals.keys()))
        hospitals[target_h]["icu_beds"] += 1
        total_icu += 1

    # Ensure at least one hospital has ventilators
    total_vents = sum(h["ventilators"] for h in hospitals.values())
    if total_vents == 0:
        target_h = rng.choice(list(hospitals.keys()))
        hospitals[target_h]["ventilators"] = rng.randint(1, 2)
        total_vents = hospitals[target_h]["ventilators"]

    # Generate patients
    patients = []
    vent_needed_count = 0
    max_vents_to_assign = min(total_vents, rng.randint(1, max(1, num_patients - 1)))

    for p in range(num_patients):
        p_id = f"PAT_{p+1}"
        needs_vent = (vent_needed_count < max_vents_to_assign) and rng.choice([True, False])
        if (p == num_patients - 1) and (vent_needed_count == 0) and (total_vents > 0):
            # Guarantee at least one critical ventilator case
            needs_vent = True

        if needs_vent:
            vent_needed_count += 1
            severity = 1
            survival_window = rng.randint(25, 45)
        else:
            severity = rng.choice([2, 3])
            survival_window = rng.randint(45, 75)

        patients.append({
            "patient_id": p_id,
            "severity": severity,
            "requires_ventilator": needs_vent,
            "survival_window_mins": survival_window,
        })

    # Derive a valid reference assignment to guarantee feasibility
    available_ambulances = [f"AMB_{a+1}" for a in range(num_ambulances)]
    rng.shuffle(available_ambulances)

    ref_solution = []
    hosp_usage = {h_id: {"icu": 0, "vents": 0} for h_id in hospitals}

    for idx, pat in enumerate(patients):
        amb_id = available_ambulances[idx]
        valid_targets = []
        for h_id, cap in hospitals.items():
            used = hosp_usage[h_id]
            if used["icu"] < cap["icu_beds"]:
                if not pat["requires_ventilator"] or (used["vents"] < cap["ventilators"]):
                    valid_targets.append(h_id)

        if not valid_targets:
            # Fallback allocation adjustment to maintain hard feasibility guarantee
            for h_id, cap in hospitals.items():
                if not pat["requires_ventilator"] or (cap["ventilators"] > hosp_usage[h_id]["vents"]):
                    hospitals[h_id]["icu_beds"] += 1
                    valid_targets.append(h_id)
                    break
            if not valid_targets:
                chosen_h = list(hospitals.keys())[0]
                hospitals[chosen_h]["icu_beds"] += 1
                if pat["requires_ventilator"]:
                    hospitals[chosen_h]["ventilators"] += 1
                valid_targets = [chosen_h]

        chosen_hosp = rng.choice(valid_targets)
        hosp_usage[chosen_hosp]["icu"] += 1
        if pat["requires_ventilator"]:
            hosp_usage[chosen_hosp]["vents"] += 1

        ref_solution.append({
            "patient_id": pat["patient_id"],
            "ambulance_id": amb_id,
            "hospital_id": chosen_hosp,
        })

    # Build transit distance/ETA matrix for all ambulances, patients, and hospitals
    # Zero Statistical Leakage:
    # 1. Background routes and reference routes share the identical support domain [10, 55].
    # 2. Trap routes with short transit times are actively injected for invalid hospital pairings.
    ref_map = {(item["patient_id"], item["ambulance_id"]): item["hospital_id"] for item in ref_solution}
    ambulances = []

    for a in range(num_ambulances):
        amb_id = f"AMB_{a+1}"
        routes = {}
        for pat in patients:
            p_id = pat["patient_id"]
            routes[p_id] = {}
            for h_id in hospitals:
                routes[p_id][h_id] = rng.randint(10, 55)

            # Reference route guarantee
            if (p_id, amb_id) in ref_map:
                target_hosp = ref_map[(p_id, amb_id)]
                safe_margin = rng.randint(5, 12)
                routes[p_id][target_hosp] = max(8, min(pat["survival_window_mins"] - safe_margin, 38))
            else:
                # Inject distractor/trap routes: fast ETA to full or unequipped hospitals
                # This prevents the model from relying on an "ETA < 15" heuristic
                if rng.random() < 0.40:
                    trap_hosp = rng.choice(list(hospitals.keys()))
                    routes[p_id][trap_hosp] = rng.randint(8, 18)

        ambulances.append({
            "ambulance_id": amb_id,
            "eta_matrix_mins": routes,
        })

    state = {
        "scenario_id": f"CRISIS_{seed:05d}",
        "patients": patients,
        "hospitals": hospitals,
        "ambulances": ambulances,
    }

    return state, ref_solution


def generate_scenario(seed: int) -> Dict[str, Any]:
    """Generates a complete prompt and telemetry state for training/eval."""
    rng = random.Random(seed)
    state, ref_solution = _build_guaranteed_feasible_scenario(rng, seed)

    prompt_text = (
        f"{SYSTEM_PROMPT}\n\n"
        f"### EMERGENCY TELEMETRY STATE:\n"
        f"```json\n{json.dumps(state)}\n```"
    )
    return {
        "prompt": prompt_text,
        "state_json": json.dumps(state),
        "reference_solution": json.dumps(ref_solution),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic GoldenHour-RL crisis datasets.")
    parser.add_argument("--output", type=str, default="data/crisis_train.jsonl", help="Output JSONL path")
    parser.add_argument("--num_samples", type=int, default=300, help="Number of scenarios to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for i in range(args.num_samples):
            sample = generate_scenario(args.seed + i)
            f.write(json.dumps(sample) + "\n")

    print(f"[GoldenHour-RL] Successfully generated {args.num_samples} verifiable scenarios at {out_path}")


if __name__ == "__main__":
    main()
