import json
import random
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
from goldenhour_rl.data.crisis_generator import _build_guaranteed_feasible_scenario, SYSTEM_PROMPT
from goldenhour_rl.sandbox.executor import execute_dispatch_code
from goldenhour_rl.sandbox.verifier import verify_dispatch_plan


def generate_algorithmic_solver_code() -> str:
    """Generates clean, verifiable dynamic dispatch logic in Python."""
    return (
        "def get_dispatch_plan(state):\n"
        "    patients = state.get('patients', [])\n"
        "    ambulances = state.get('ambulances', [])\n"
        "    hospitals = state.get('hospitals', {})\n"
        "    \n"
        "    # Priority order: ventilator-dependent first, then shortest survival window\n"
        "    sorted_patients = sorted(\n"
        "        patients,\n"
        "        key=lambda p: (not p.get('requires_ventilator', False), p.get('survival_window_mins', 999))\n"
        "    )\n"
        "    \n"
        "    used_ambulances = set()\n"
        "    hospital_usage = {h_id: {'icu': 0, 'vent': 0} for h_id in hospitals}\n"
        "    plan = []\n"
        "    \n"
        "    for pat in sorted_patients:\n"
        "        p_id = pat['patient_id']\n"
        "        needs_vent = pat.get('requires_ventilator', False)\n"
        "        deadline = pat.get('survival_window_mins', 999)\n"
        "        \n"
        "        best_amb = None\n"
        "        best_hosp = None\n"
        "        min_eta = 9999\n"
        "        \n"
        "        for amb in ambulances:\n"
        "            a_id = amb['ambulance_id']\n"
        "            if a_id in used_ambulances:\n"
        "                continue\n"
        "            eta_table = amb.get('eta_matrix_mins', {}).get(p_id, {})\n"
        "            \n"
        "            for h_id, caps in hospitals.items():\n"
        "                if hospital_usage[h_id]['icu'] >= caps.get('icu_beds', 0):\n"
        "                    continue\n"
        "                if needs_vent and hospital_usage[h_id]['vent'] >= caps.get('ventilators', 0):\n"
        "                    continue\n"
        "                \n"
        "                eta = eta_table.get(h_id, 9999)\n"
        "                if eta <= deadline and eta < min_eta:\n"
        "                    min_eta = eta\n"
        "                    best_amb = a_id\n"
        "                    best_hosp = h_id\n"
        "        \n"
        "        if best_amb and best_hosp:\n"
        "            used_ambulances.add(best_amb)\n"
        "            hospital_usage[best_hosp]['icu'] += 1\n"
        "            if needs_vent:\n"
        "                hospital_usage[best_hosp]['vent'] += 1\n"
        "            plan.append({\n"
        "                'patient_id': p_id,\n"
        "                'ambulance_id': best_amb,\n"
        "                'hospital_id': best_hosp,\n"
        "            })\n"
        "            \n"
        "    return plan"
    )


def generate_direct_schedule_code(plan: List[Dict[str, str]]) -> str:
    """Generates exact verified schedule literal wrapped in get_dispatch_plan."""
    formatted_items = []
    for item in plan:
        formatted_items.append(
            f"        {{'patient_id': '{item['patient_id']}', 'ambulance_id': '{item['ambulance_id']}', 'hospital_id': '{item['hospital_id']}'}}"
        )
    items_str = ",\n".join(formatted_items)
    return (
        "def get_dispatch_plan(state):\n"
        "    return [\n"
        f"{items_str}\n"
        "    ]"
    )


def synthesize_reasoning_trace(state: Dict[str, Any], plan: List[Dict[str, str]]) -> str:
    """Synthesizes concise triage analysis strictly within the 150-word ceiling."""
    patients = state["patients"]
    hospitals = state["hospitals"]
    ambulances = state["ambulances"]

    crit_pats = [p for p in patients if p.get("requires_ventilator")]
    crit_ids = ", ".join(p["patient_id"] for p in crit_pats) if crit_pats else "None"

    # Compute total capacities
    total_icu = sum(h.get("icu_beds", 0) for h in hospitals.values())
    total_vents = sum(h.get("ventilators", 0) for h in hospitals.values())

    return (
        f"Assessing crisis scenario: {len(patients)} casualties, {len(ambulances)} available ambulances, "
        f"and {len(hospitals)} regional hospitals with {total_icu} ICU beds and {total_vents} ventilators total. "
        f"Priority triage: Patients requiring ventilators ({crit_ids}) are prioritized for facilities with active life support. "
        f"Dispatch allocation: Matched dedicated ambulances to patients and destination hospitals to ensure transit ETAs "
        f"remain strictly within survival windows while preventing ICU/ventilator capacity overflows and ambulance double-booking."
    )


def generate_verified_sft_sample(seed: int, prefer_algorithmic: bool = True) -> Dict[str, Any]:
    """
    Constructs a verified crisis scenario and pairs it with an expert reasoning trace
    and a provably 100% invariant-compliant Python dispatch implementation.
    """
    rng = random.Random(seed)
    state, ref_solution = _build_guaranteed_feasible_scenario(rng, seed)
    state_json = json.dumps(state)

    chosen_code = None
    chosen_plan = None

    if prefer_algorithmic:
        algo_code = generate_algorithmic_solver_code()
        exec_res = execute_dispatch_code(algo_code, timeout_sec=2.0, state_str=state_json)
        if exec_res.get("ok") and isinstance(exec_res.get("plan"), list):
            ver_res = verify_dispatch_plan(exec_res["plan"], state_json)
            if ver_res["valid"]:
                chosen_code = algo_code
                chosen_plan = exec_res["plan"]

    if chosen_code is None:
        # Fallback to mathematically guaranteed reference solution
        chosen_code = generate_direct_schedule_code(ref_solution)
        chosen_plan = ref_solution

    # Final invariant sanity check
    final_ver = verify_dispatch_plan(chosen_plan, state_json)
    assert final_ver["valid"], f"Seed {seed} failed verification: {final_ver['violations']}"

    think_trace = synthesize_reasoning_trace(state, chosen_plan)
    completion_text = f"<think>\n{think_trace}\n</think>\n```python\n{chosen_code}\n```"

    prompt_text = (
        f"{SYSTEM_PROMPT}\n\n"
        f"### EMERGENCY TELEMETRY STATE:\n"
        f"```json\n{state_json}\n```"
    )

    messages = [
        {"role": "user", "content": prompt_text},
        {"role": "assistant", "content": completion_text},
    ]

    return {
        "prompt": prompt_text,
        "messages": messages,
        "completion": completion_text,
        "state_json": state_json,
        "reference_solution": json.dumps(chosen_plan),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate verified Cold-Start SFT dataset for GoldenHour-RL.")
    parser.add_argument("--output", type=str, default="data/crisis_sft.jsonl", help="Output path for SFT dataset")
    parser.add_argument("--num_samples", type=int, default=300, help="Number of verified demonstrations")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--algorithmic_ratio", type=float, default=0.70, help="Fraction of algorithmic solver demonstrations")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    verified_count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i in range(args.num_samples):
            use_algo = (i / args.num_samples) < args.algorithmic_ratio
            sample = generate_verified_sft_sample(args.seed + i, prefer_algorithmic=use_algo)
            f.write(json.dumps(sample) + "\n")
            verified_count += 1

    print(f"[GoldenHour-RL SFT] Successfully generated and verified {verified_count}/{args.num_samples} demonstrations at {out_path}")


if __name__ == "__main__":
    main()
