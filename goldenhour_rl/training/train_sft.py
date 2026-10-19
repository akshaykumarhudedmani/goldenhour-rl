import os
import argparse
import torch
from pathlib import Path
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()


def run_mock_cpu_sft(args):
    """Local CPU dry-run verification mode for SFT pipeline."""
    print("[GoldenHour-RL SFT] Running in --mock_cpu verification mode...")
    dataset = load_dataset("json", data_files=args.data_path, split="train")
    print(f"[GoldenHour-RL SFT] Successfully loaded dataset: {len(dataset)} samples.")
    sample = dataset[0]
    assert "messages" in sample, "Dataset sample must contain 'messages' field."
    assert len(sample["messages"]) == 2, "Sample messages must contain user and assistant turns."
    print(f"[GoldenHour-RL SFT] First sample user prompt length: {len(sample['messages'][0]['content'])} chars")
    print(f"[GoldenHour-RL SFT] First sample assistant completion length: {len(sample['messages'][1]['content'])} chars")
    print("[GoldenHour-RL SFT] Mock CPU SFT pipeline completed successfully!")


def run_sft_training(args):
    if args.mock_cpu or not torch.cuda.is_available():
        if not torch.cuda.is_available() and not args.mock_cpu:
            print("[GoldenHour-RL SFT] CUDA GPU not detected. Falling back to --mock_cpu dry-run mode.")
        run_mock_cpu_sft(args)
        return

    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig

    print(f"[GoldenHour-RL SFT] Initializing Unsloth 4-bit model: {args.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=2048,
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

    dataset = load_dataset("json", data_files=args.data_path, split="train")

    def formatting_prompts_func(batch):
        formatted = []
        for convo in batch["messages"]:
            text = tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
            formatted.append(text)
        return {"text": formatted}

    dataset = dataset.map(formatting_prompts_func, batched=True)

    training_args = SFTConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=args.epochs,
        logging_steps=5,
        save_strategy="epoch",
        dataset_text_field="text",
        max_seq_length=2048,
        report_to="wandb" if args.use_wandb else "none",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset,
    )

    torch.cuda.reset_peak_memory_stats()
    print("[GoldenHour-RL SFT] Commencing Cold-Start Supervised Fine-Tuning loop...")
    trainer.train()

    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"\n[GoldenHour-RL SFT Telemetry] Peak Training VRAM: {peak_vram_gb:.2f} GB")

    out_lora = Path(args.output_dir) / "final_lora"
    out_lora.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_lora))
    tokenizer.save_pretrained(str(out_lora))
    print(f"[GoldenHour-RL SFT] Trained SFT adapter saved to {out_lora}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GoldenHour-RL Cold-Start SFT Trainer")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--data_path", type=str, default="data/crisis_sft.jsonl")
    parser.add_argument("--output_dir", type=str, default="outputs/goldenhour_sft")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--mock_cpu", action="store_true", help="Run local CPU dry-run without GPU")
    args = parser.parse_args()
    run_sft_training(args)
