import json
import os
import sys
import time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from goldenhour_rl.eval.benchmark import extract_python_code
from goldenhour_rl.sandbox.executor import execute_dispatch_code
from goldenhour_rl.sandbox.verifier import verify_dispatch_plan

def run_evaluation(model_path="outputs/goldenhour_grpo/final_lora", eval_file="data/crisis_eval.jsonl", max_samples=50):
    print("=" * 80)
    print("   GOLDENHOUR-RL v0.3: OOD BENCHMARK EVALUATION (Pass@1 & Best-of-4)")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)

    base_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    print(f"[Loading] Base: {base_name} | Adapter: {model_path} on {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
        )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.bfloat16,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    if device == "cpu":
        base_model = base_model.to(device)

    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()

    with open(eval_file, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()][:max_samples]

    print(f"[Dataset] Loaded {len(samples)} crisis scenarios from {eval_file}")

    eos_ids = [tokenizer.eos_token_id]
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end is not None and im_end not in eos_ids:
        eos_ids.append(im_end)

    for mode, num_tries, temp in [("Pass@1", 1, 0.2), ("Best-of-4", 4, 0.7)]:
        print(f"\n---> Commencing {mode} (T={temp}, tries={num_tries}) across {len(samples)} scenarios...")
        exec_count, zero_viol, total_pats, surv_strict, surv_lenient = 0, 0, 0, 0, 0
        icu_ov, vent_ov, total_viols = 0, 0, 0

        t0 = time.time()
        for i, s in enumerate(samples):
            state = json.loads(s["state_json"])
            n_p = len(state["patients"])
            total_pats += n_p

            prompt = s["prompt"]
            fmt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
            inp = tokenizer(fmt, return_tensors="pt").to(device)

            gen_kwargs = {
                "max_new_tokens": 650,
                "do_sample": (temp > 0.0),
                "pad_token_id": tokenizer.eos_token_id,
                "eos_token_id": eos_ids,
            }
            if temp > 0.0:
                gen_kwargs["temperature"] = temp
                gen_kwargs["top_p"] = 0.95
                gen_kwargs["top_k"] = 50
            if num_tries > 1:
                gen_kwargs["num_return_sequences"] = num_tries

            with torch.no_grad():
                out = model.generate(**inp, **gen_kwargs)

            prompt_len = inp.input_ids.shape[1]
            best_ver = None
            sample_exec = False

            for seq in out:
                text = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
                code = extract_python_code(text)
                if not code:
                    continue
                ex = execute_dispatch_code(code, state_str=s["state_json"])
                if not ex.get("ok") or not isinstance(ex.get("plan"), list):
                    continue
                sample_exec = True
                ver = verify_dispatch_plan(ex["plan"], s["state_json"])
                if best_ver is None or len(ver["violations"]) < len(best_ver["violations"]):
                    best_ver = ver
                if ver["valid"]:
                    break

            if sample_exec:
                exec_count += 1

            if best_ver is not None:
                if best_ver["valid"]:
                    zero_viol += 1
                total_viols += len(best_ver["violations"])
                stats = best_ver.get("stats", {})
                icu_ov += stats.get("icu_overflow_count", 0)
                vent_ov += stats.get("vent_overflow_count", 0)
                surv_strict += stats.get("survived_count", 0)
                surv_lenient += max(0, n_p - stats.get("breached_count", 0) - stats.get("unassigned_count", 0))
            else:
                total_viols += n_p

            if (i + 1) % 5 == 0 or (i + 1) == len(samples):
                elapsed = time.time() - t0
                print(f"  [{mode}] Progress: {i+1}/{len(samples)} in {elapsed:.1f}s | Exec: {exec_count}/{i+1} | Zero-Viol: {zero_viol}/{i+1}", flush=True)

        n = len(samples)
        exec_rate = round(100.0 * exec_count / n, 1)
        zv_rate = round(100.0 * zero_viol / n, 1)
        strict_rate = round(100.0 * surv_strict / max(1, total_pats), 1)
        lenient_rate = round(100.0 * surv_lenient / max(1, total_pats), 1)

        print(f"\n=== RESULTS FOR {mode} ===")
        print(f"Executable Plan % : {exec_rate}%")
        print(f"Zero-Violation %  : {zv_rate}%")
        print(f"Strict Survival % : {strict_rate}%")
        print(f"Lenient Survival %: {lenient_rate}%")
        print(f"ICU Overflows     : {icu_ov}")
        print(f"Vent Overflows    : {vent_ov}")
        print(f"Total Violations  : {total_viols}")
        print("-" * 50)

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    run_evaluation(max_samples=n)
