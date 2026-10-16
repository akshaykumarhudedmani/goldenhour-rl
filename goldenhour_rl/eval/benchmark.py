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


def extract_python_code(text: str) -> str | None:
    """Multi-fallback code extractor for resilient markdown parsing."""
    import re
    # Fallback 1: Standard markdown code blocks containing def get_dispatch_plan
    matches = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL)
    for m in matches:
        if "def get_dispatch_plan" in m:
            return m.strip()

    # Fallback 2: Unclosed markdown code block
    unclosed = re.search(r"```(?:python)?\s*(.*?)$", text, flags=re.DOTALL)
    if unclosed and "def get_dispatch_plan" in unclosed.group(1):
        return unclosed.group(1).strip()

    # Fallback 3: Raw text starting from optional imports and def get_dispatch_plan
    raw_match = re.search(r"((?:import\s+[^\n]+\n+)*def get_dispatch_plan\s*\(.*)", text, flags=re.DOTALL)
    if raw_match:
        return raw_match.group(1).split("```")[0].strip()

    # Fallback 4: Raw list literal inside markdown or text containing patient_id
    raw_list = re.search(r"(\[\s*\{\s*['\"]patient_id['\"].*?\])", text, flags=re.DOTALL)
    if raw_list:
        return f"def get_dispatch_plan():\n    return {raw_list.group(1).strip()}"

    return None


def evaluate_model_on_dataset(
    dataset_path: str,
    lora_path: str,
    base_model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    max_samples: int = 50,
) -> Dict[str, Any]:
    """Evaluates an actual trained LoRA adapter on the crisis dataset."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from goldenhour_rl.sandbox.executor import execute_dispatch_code

    print(f"\n[GoldenHour-RL Benchmark] Loading base model '{base_model_name}' and adapter '{lora_path}'...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(lora_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    if device == "cpu":
        base_model = base_model.to(device)

    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()

    samples = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= max_samples:
                break
            if line.strip():
                samples.append(json.loads(line))

    executable_count = 0
    zero_violation_count = 0
    total_violations = 0
    total_sla_margins = 0.0
    total_patients = 0
    total_breaches = 0
    total_icu_overflows = 0
    total_vent_overflows = 0

    print(f"[GoldenHour-RL Benchmark] Running inference across {len(samples)} scenarios on {device}...")
    for i, sample in enumerate(samples):
        prompt = sample["prompt"]
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.6,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        code_str = extract_python_code(generated_text)
        if code_str:
            exec_res = execute_dispatch_code(code_str, timeout_sec=2.0, state_str=sample["state_json"])
            if exec_res.get("ok"):
                executable_count += 1
                plan = exec_res.get("plan", [])
            else:
                plan = []
        else:
            plan = []

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

    total_scenarios = len(samples)
    executable_rate = (executable_count / total_scenarios) * 100 if total_scenarios > 0 else 0.0
    pass_rate = (zero_violation_count / total_scenarios) * 100 if total_scenarios > 0 else 0.0
    avg_margin = (total_sla_margins / zero_violation_count) if zero_violation_count > 0 else 0.0
    patient_survival_rate = ((total_patients - total_breaches) / total_patients * 100) if total_patients > 0 else 0.0

    return {
        "strategy": f"GoldenHour-RL ({Path(lora_path).name})",
        "scenarios": total_scenarios,
        "executable_plan_rate": round(executable_rate, 2),
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
    parser.add_argument("--lora_path", type=str, default=None, help="Path or HuggingFace repo ID of trained LoRA adapter")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--max_samples", type=int, default=50, help="Max scenarios to evaluate")
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

    if args.lora_path:
        model_res = evaluate_model_on_dataset(
            str(eval_path),
            args.lora_path,
            base_model_name=args.base_model,
            max_samples=args.max_samples
        )
        results.append(model_res)

    header = f"{'Strategy':<20} | {'Executable %':<14} | {'Zero-Viol %':<12} | {'Survival %':<12} | {'ICU Overflows':<14} | {'Vent Overflows':<14} | {'Violations'}"
    print(header)
    print("-" * len(header))

    for r in results:
        exec_rate = f"{r.get('executable_plan_rate', 100.0):.1f}%"
        line = (
            f"{r['strategy']:<20} | "
            f"{exec_rate:<14} | "
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
