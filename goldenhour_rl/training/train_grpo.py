import os
import argparse
import torch
from pathlib import Path
from datasets import load_dataset
from dotenv import load_dotenv

# Automatically load environment variables (.env)
load_dotenv()

from goldenhour_rl.training.rewards import format_reward_func, verifier_reward_func


def run_mock_cpu_pipeline(args):
    """
    Dry-run test mode for local CPU development.
    Validates dataset loading, sample parsing, and reward pipeline without requiring a GPU.
    """
    print("[GoldenHour-RL] Running in --mock_cpu verification mode...")
    dataset = load_dataset("json", data_files=args.data_path, split="train")
    print(f"[GoldenHour-RL] Successfully loaded dataset: {len(dataset)} samples.")

    # Select first sample to dry-run reward functions
    sample = dataset[0]
    mock_completion = (
        "<think>Triage analysis: Patient PAT_1 requires a ventilator. Hospital HOSP_1 has capacity. "
        "Assigning AMB_1 to reach within survival deadline.</think>\n"
        "```python\n"
        "def get_dispatch_plan():\n"
        "    return []\n"
        "```"
    )

    fmt_r = format_reward_func([sample["prompt"]], [mock_completion])
    ver_r = verifier_reward_func([sample["prompt"]], [mock_completion], state_json=[sample["state_json"]])

    print(f"[GoldenHour-RL] Mock Format Reward: {fmt_r[0]:.2f}")
    print(f"[GoldenHour-RL] Mock Verifier Reward: {ver_r[0]:.2f}")
    print("[GoldenHour-RL] Mock CPU pipeline dry-run completed successfully!")


def run_training(args):
    if args.mock_cpu or not torch.cuda.is_available():
        if not torch.cuda.is_available() and not args.mock_cpu:
            print("[GoldenHour-RL] CUDA GPU not detected. Falling back to --mock_cpu dry-run mode.")
        run_mock_cpu_pipeline(args)
        return

    # Cloud GPU Training Loop (Unsloth + TRL GRPOTrainer)
    from unsloth import FastLanguageModel
    from peft import PeftModel
    from trl import GRPOConfig, GRPOTrainer

    print(f"[GoldenHour-RL] Initializing Unsloth 4-bit model: {args.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=4096,
        load_in_4bit=True,
        fast_inference=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    if args.init_lora_path and Path(args.init_lora_path).exists():
        print(f"[GoldenHour-RL] Initializing policy weights from Cold-Start SFT LoRA: {args.init_lora_path}")
        try:
            from peft import load_peft_weights, set_peft_model_state_dict
            sft_weights = load_peft_weights(args.init_lora_path)
            set_peft_model_state_dict(model, sft_weights)
            print("[GoldenHour-RL] Successfully transferred SFT LoRA weights into trainable GRPO model!")
        except Exception as e:
            print(f"[GoldenHour-RL] Direct weight injection notice ({e}), loading via PeftModel fallback...")
            model = PeftModel.from_pretrained(model, args.init_lora_path, is_trainable=True)

    # Sanity-check trainable parameter health to prevent zero-gradient dead runs
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"[GoldenHour-RL Telemetry] Trainable parameters: {trainable_params:,} / {all_params:,} ({100 * trainable_params / all_params:.2f}%)")
    assert trainable_params > 0, "FATAL: Zero trainable parameters detected! Check LoRA adapter initialization."

    dataset = load_dataset("json", data_files=args.data_path, split="train")

    def ensure_conversational_prompt(example):
        """Ensures prompt is formatted as conversational messages so GRPOTrainer applies ChatML templates."""
        if "messages" in example and isinstance(example["messages"], list):
            user_msgs = [m for m in example["messages"] if m.get("role") in ("system", "user")]
            return {"prompt": user_msgs}
        elif isinstance(example.get("prompt"), str):
            return {"prompt": [{"role": "user", "content": example["prompt"]}]}
        return example

    dataset = dataset.map(ensure_conversational_prompt)

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=5e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=args.num_generations,
        max_prompt_length=2048,
        max_completion_length=1536,
        max_steps=args.max_steps,
        logging_steps=5,
        save_steps=50,
        report_to="wandb" if args.use_wandb else "none",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[format_reward_func, verifier_reward_func],
        args=training_args,
        train_dataset=dataset,
    )

    torch.cuda.reset_peak_memory_stats()
    print("[GoldenHour-RL] Commencing GRPO reinforcement learning training loop...")
    trainer.train()

    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"\n[GoldenHour-RL Telemetry] Peak Training VRAM: {peak_vram_gb:.2f} GB")

    out_lora = Path(args.output_dir) / "final_lora"
    out_lora.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_lora))
    tokenizer.save_pretrained(str(out_lora))
    print(f"[GoldenHour-RL] LoRA weights and tokenizer saved successfully to {out_lora}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GoldenHour-RL GRPO Training Harness")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--init_lora_path", type=str, default=None, help="Path to initial Cold-Start SFT LoRA checkpoint")
    parser.add_argument("--data_path", type=str, default="data/crisis_train.jsonl")
    parser.add_argument("--output_dir", type=str, default="outputs/goldenhour_grpo")
    parser.add_argument("--max_steps", type=int, default=150)
    parser.add_argument("--num_generations", type=int, default=8, help="Number of completions per prompt rollout (default: 8)")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--mock_cpu", action="store_true", help="Run local CPU dry-run without GPU")
    args = parser.parse_args()
    run_training(args)
