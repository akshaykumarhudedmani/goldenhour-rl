import json
import os
import gradio as gr
from goldenhour_rl.data.crisis_generator import generate_scenario
from goldenhour_rl.sandbox.verifier import verify_dispatch_plan
from goldenhour_rl.eval.benchmark import greedy_nearest_heuristic, extract_python_code
from goldenhour_rl.sandbox.executor import execute_dispatch_code

# --- GoldenHour-RL Model Weights Binding ---
MODEL_REPO_ID = "akshaykumarhudedmani/goldenhour-rl"
BASE_MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

_cached_model = None
_cached_tokenizer = None


def get_model():
    """Lazily loads the fine-tuned GoldenHour-RL GRPO model from Hugging Face Hub."""
    global _cached_model, _cached_tokenizer
    if _cached_model is None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel

            print(f"[Spaces] Loading tokenizer & LoRA adapter from {MODEL_REPO_ID}...")
            _cached_tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO_ID, trust_remote_code=True)
            base = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_NAME,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )
            _cached_model = PeftModel.from_pretrained(base, MODEL_REPO_ID)
            _cached_model.eval()
            print("[Spaces] Model loaded successfully!")
        except Exception as e:
            print(f"[Spaces Note] Running high-speed sandbox simulation mode: {e}")
    return _cached_model, _cached_tokenizer


def format_plan_table(plan, state):
    patients = {p["patient_id"]: p for p in state["patients"]}
    ambulances = {a["ambulance_id"]: a["eta_matrix_mins"] for a in state["ambulances"]}

    rows = []
    for item in plan:
        p_id = item.get("patient_id")
        a_id = item.get("ambulance_id")
        h_id = item.get("hospital_id")
        pat = patients.get(p_id, {})
        eta = ambulances.get(a_id, {}).get(p_id, {}).get(h_id, "N/A")
        window = pat.get("survival_window_mins", "N/A")
        vent = "Required" if pat.get("requires_ventilator") else "No"
        rows.append([p_id, f"Sev {pat.get('severity')}", vent, a_id, h_id, f"{eta} min", f"{window} min"])
    return rows


def run_simulation(seed, engine_mode="Synthetic Deterministic Ground-Truth Baseline"):
    sample = generate_scenario(int(seed))
    state = json.loads(sample["state_json"])

    # 1. Base Naive / Greedy Dispatch (Simulates unconstrained LLM greedily optimizing for nearest hospital)
    base_plan = greedy_nearest_heuristic(state)
    base_ver = verify_dispatch_plan(base_plan, sample["state_json"])

    # 2. Target Dispatch Policy (Synthetic Deterministic Ground-Truth Baseline vs Live SLM Inference)
    rl_plan = None
    model_source = "Synthetic Deterministic Ground-Truth Baseline"
    fallback_note = ""

    if engine_mode == "Live SLM Inference (Requires GPU)":
        model, tokenizer = get_model()
        if model is not None and tokenizer is not None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                messages = [{"role": "user", "content": sample["prompt"]}]
                fmt_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(fmt_prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
                gen_text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                code = extract_python_code(gen_text)
                if code:
                    ex = execute_dispatch_code(code, state_str=sample["state_json"])
                    if ex["ok"] and isinstance(ex.get("plan"), list):
                        rl_plan = ex["plan"]
                        model_source = f"Live SLM Inference: GoldenHour-RL GRPO LoRA ({MODEL_REPO_ID})"
                    else:
                        fallback_note = f" (Execution failed: {ex.get('error', 'Malformed plan')})"
            except Exception as e:
                fallback_note = f" (Inference exception: {str(e)})"
        else:
            fallback_note = " (Weights could not be initialized on host CPU/RAM)"

    if rl_plan is None:
        rl_plan = json.loads(sample.get("reference_solution", "[]"))
        if engine_mode == "Live SLM Inference (Requires GPU)":
            model_source = f"Synthetic Deterministic Ground-Truth Baseline [Fallback - Live SLM unavailable{fallback_note}]"
        else:
            model_source = "Synthetic Deterministic Ground-Truth Baseline (Oracle Reference from Synthesizer)"

    rl_ver = verify_dispatch_plan(rl_plan, sample["state_json"])

    # Format telemetry outputs
    base_status = "CRITICAL VIOLATIONS DETECTED" if not base_ver["valid"] else "Valid"
    rl_status = "100% VERIFIED ZERO VIOLATIONS" if rl_ver["valid"] else "Failed"

    base_viol_text = "\n".join([f"• {v}" for v in base_ver["violations"]]) if base_ver["violations"] else "None (All invariants passed)"
    rl_viol_text = f"Plan Source: {model_source}\nInvariant Verification: ICU beds, ventilators, single-ambulance, SLA deadlines checked."

    base_rows = format_plan_table(base_plan, state)
    rl_rows = format_plan_table(rl_plan, state)

    telemetry_state_str = json.dumps(state, indent=2)

    return (
        telemetry_state_str,
        base_status,
        base_viol_text,
        base_rows,
        rl_status,
        rl_viol_text,
        f"{rl_ver['sla_score'] * 100:.1f}% Avg SLA Margin",
        rl_rows,
    )


with gr.Blocks(title="GoldenHour-RL: Verifiable Crisis Dispatch Telemetry") as demo:
    gr.Markdown("# ⏱️ Project GoldenHour-RL: Crisis Dispatch Telemetry Benchmark")
    gr.Markdown(
        "Empirical evaluation testbed analyzing whether sub-3B Small Language Models can synthesize verifiable dispatch "
        "logic under tight resource budgets ($\le 8\\text{ GB}$ VRAM). Compares unconstrained nearest-neighbor dispatch "
        "(causing ventilator shortages and ICU bed overflow) against verifiable constraint-satisfaction schedules.\n\n"
        f"**Model Adapter Binding:** [`{MODEL_REPO_ID}`](https://huggingface.co/{MODEL_REPO_ID}) (Release v0.3.0 • 50 Steps • +2.23 Verifier Reward)"
    )

    with gr.Row():
        seed_input = gr.Number(value=42, label="Scenario Random Seed", precision=0)
        mode_selector = gr.Radio(
            choices=["Synthetic Deterministic Ground-Truth Baseline", "Live SLM Inference (Requires GPU)"],
            value="Synthetic Deterministic Ground-Truth Baseline",
            label="Dispatch Execution Engine"
        )
        simulate_btn = gr.Button("⚡ Simulate Emergency Crisis", variant="primary")

    with gr.Accordion("🔍 Live Emergency Telemetry State (Patients, Ambulances, Hospitals)", open=False):
        state_view = gr.Code(language="json", label="Crisis State JSON")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### ❌ Unconstrained Dispatch (Naive Nearest Heuristic)")
            gr.Markdown("*Greedily routes to nearest hospital without global bottleneck verification.*")
            base_status_out = gr.Textbox(label="Invariant Safety Status", value="")
            base_viol_out = gr.Textbox(label="Life-Support Invariant Violations", lines=4)
            base_table = gr.Dataframe(
                headers=["Patient", "Severity", "Ventilator", "Ambulance", "Hospital", "ETA", "SLA Deadline"],
                label="Generated Dispatch Schedule"
            )

        with gr.Column():
            gr.Markdown("### 📋 Evaluated Dispatch Schedule")
            gr.Markdown("*Displays either the Synthetic Ground-Truth Reference or live Qwen2.5-Coder-1.5B LoRA generation.*")
            rl_status_out = gr.Textbox(label="Invariant Safety Status", value="")
            rl_margin_out = gr.Textbox(label="Golden-Hour Survival Margin", value="")
            rl_viol_out = gr.Textbox(label="Execution Engine & Dispatch Source", lines=2)
            rl_table = gr.Dataframe(
                headers=["Patient", "Severity", "Ventilator", "Ambulance", "Hospital", "ETA", "SLA Deadline"],
                label="Dispatch Schedule Table"
            )

    simulate_btn.click(
        fn=run_simulation,
        inputs=[seed_input, mode_selector],
        outputs=[
            state_view,
            base_status_out,
            base_viol_out,
            base_table,
            rl_status_out,
            rl_viol_out,
            rl_margin_out,
            rl_table,
        ]
    )

if __name__ == "__main__":
    demo.launch()
