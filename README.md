# Project GoldenHour-RL: Verifiable Crisis Dispatch RL Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![RL: GRPO](https://img.shields.io/badge/RL-GRPO%20(TRL%20%2B%20Unsloth)-red.svg)](https://github.com/unslothai/unsloth)
[![Cost: $0.00](https://img.shields.io/badge/Cost-%240.00%20Zero--Cost-brightgreen.svg)]()

> **"During the 'Golden Hour'—the first 60 minutes after critical trauma or disaster surge—survival drops by nearly 10% for every 6 minutes of delay. In a chatbot, a hallucination is an annoyance. In the Golden Hour, a hallucination costs a life."**

---

## 1. Problem Statement & Systems Architecture

Standard LLMs hallucinate when asked to coordinate emergency medical logistics: they route critical patients to hospitals with zero available ventilators, double-book ambulances, or miscalculate transit windows.

**GoldenHour-RL** is an open-source, verifiable Reinforcement Learning (RL) post-training framework. It trains lightweight reasoning models (`Qwen/Qwen2.5-Coder-1.5B-Instruct`) via **Group Relative Policy Optimization (GRPO)** inside a deterministic, zero-trust crisis simulation sandbox.

```
                           +-------------------------------------+
                           |  Emergency Telemetry State (JSON)   |
                           |  (Patients, Ambulances, Hospitals)  |
                           +-------------------------------------+
                                              |
                                              v
                           +-------------------------------------+
                           |   Qwen2.5-Coder-1.5B-Instruct       |
                           |   <think> Bottleneck Reasoning </think>
                           |   ```python get_dispatch_plan() ``` |
                           +-------------------------------------+
                                              |
                                              v
                           +-------------------------------------+
                           | Deterministic Sandbox Verifier (<2s)|
                           | - AST Security & Infinite Loop Guard|
                           | - Zero Ambulance Double-Booking     |
                           | - ICU Bed & Ventilator Limits       |
                           | - Golden-Hour SLA Transit Window    |
                           +-------------------------------------+
                                              |
                       +----------------------+----------------------+
                       |                                             |
            [100% Invariant Compliant]                     [Constraint Violated]
                       |                                             |
                       v                                             v
          Reward: +2.0 + 0.5 * SLA_Margin                Reward: -0.2 to -1.4 (Shaped)
```

### Why GRPO over PPO?
* **Eliminates the Critic Network:** Traditional Actor-Critic PPO requires holding a separate value model in GPU VRAM. GRPO computes advantages relative to group completions ($G=6$), cutting training VRAM by **>67%** (measured **5.27 GB peak VRAM** on Kaggle Tesla T4) vs ~16+ GB for Actor-Critic PPO.
* **Deterministic Life-Support Invariants:** Replaces noisy LLM-as-a-judge with mathematical ground-truth verification.
* **Trained LoRA Weights:** [Hugging Face Hub `akshaykumarhudedmani/goldenhour-rl`](https://huggingface.co/akshaykumarhudedmani/goldenhour-rl)
* **Live Experiment Telemetry:** [Weights & Biases Dashboard `goldenhour-rl`](https://wandb.ai/akshaykumarhudedmani-rl/goldenhour-rl)

---

## 2. Benchmark Telemetry: Heuristic vs. Verifiable Reference

Evaluated across $N=100$ out-of-distribution crisis scenarios:

| Strategy | Zero-Violation Rate | Patient Survival (SLA) | ICU Bed Overflows | Ventilator Overflows | Total Violations |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Greedy Nearest Hospital** | 37.0% | 91.3% | 62 | 13 | 109 |
| **GoldenHour-RL Reference** | **100.0%** | **100.0%** | **0** | **0** | **0** |

---

## 3. Empirical Training Telemetry & Systems Profiling (Weights & Biases)

Live telemetry captured from run [`brisk-waterfall-3` (`lja961ve`)](https://wandb.ai/akshaykumarhudedmani-rl/goldenhour-rl/runs/lja961ve) across 50+ GRPO optimization steps on dual Tesla T4 GPUs:

### Pillar 1: Multiprocessing Sandbox vs. Autoregressive Generation (>450x Speedup)
![Multiprocessing Sandbox vs Generation](assets/image_f16a0d.png)
* **Deterministic Verification:** The multiprocessing AST sandbox evaluates all 24 generated dispatch plans in **$<0.72$ seconds**, compared to **$250–350$ seconds** required for autoregressive token decoding on a Tesla T4.
* **Safety First:** Hard 2.0-second timeout guards and AST static analysis prevent rogue infinite loops or arbitrary system calls.

### Pillar 2: Verifier Reward Mean & Reward Spread Doubling ($\sigma: 0.105 \to 0.205$)
![Verifier Reward Progression](assets/image_f17d71.png)
* **Policy Discovery:** `train/rewards/verifier_reward_func/mean` steadily ascends toward positive territory as the model learns to respect ventilator capacities and Golden-Hour transit windows.
* **Doubled Exploration Spread:** Within-group standard deviation doubles from $\sigma = 0.105$ to $\sigma = 0.205$, confirming that policy completions differentiate into clearly superior invariant-passing plans ($A_i > 0$).

### Pillar 3: Zero Dead Batches ($\text{frac\_reward\_zero\_std} = 0.0$) & Natural Termination
![Zero Dead Batches and Terminated Length](assets/image_f17dac.png)
* **100% Productive Policy Gradients:** `train/frac_reward_zero_std` is flatlined at **$0.0$** across all steps, proving every single prompt group ($G=6$) has non-zero reward variance ($\sigma > 0$), resulting in zero wasted gradient updates.
* **Natural Reasoning Termination:** Naturally terminated completions finish with `<|im_end|>` in **$\sim 390$ tokens**, well within the safety ceiling.

### Pillar 4: Computational Parsimony — Eliminating Reasoning Bloat ($-9.2\%$ Clipped Ratio)
![Computational Parsimony](assets/image_f1806f.png)
* **Compression Dynamics:** `train/completions/clipped_ratio` drops from **$87.5\%$ down to $78.3\%$** as GRPO penalizes rambling and rewards concise $<200$-word `<think>` traces.
* **Reduced Step Latency:** Rollout generation latency accelerates by over **$26$ seconds per step** as completions become concise.

---

## 4. The $0.00 Extra-Cost Architecture

| Resource Tier | Cost | Hardware Target | Purpose |
| :--- | :--- | :--- | :--- |
| **Tier 1: Local Machine** | **$0.00** | Any CPU / 4GB+ Laptop | Scenario synthesis, deterministic sandbox verifier, and unit tests via `pytest`. |
| **Tier 2: Free Cloud GPU** | **$0.00** | Kaggle (2x T4 32GB) or Google Colab (T4 16GB) | 2-hour Unsloth 4-bit QLoRA + GRPO training loop (Peak VRAM: 5.27 GB). |
| **Tier 3: HF Space** | **$0.00** | Hugging Face Spaces | Live telemetry visualization dashboard. |

---

## 5. Quickstart & Local Setup

### Installation
```bash
git clone https://github.com/akshaykumarhudedmani/goldenhour-rl.git
cd goldenhour-rl
pip install -r requirements.txt
```

### Configure Secrets
Copy `.env.example` to `.env` (strictly ignored by `.gitignore`):
```bash
cp .env.example .env
```
Fill in your `HF_TOKEN` and `WANDB_API_KEY`.

### 1. Synthesize Crisis Telemetry Scenarios
```bash
# Generate 300 training scenarios
python -m goldenhour_rl.data.crisis_generator --output data/crisis_train.jsonl --num_samples 300 --seed 42

# Generate 100 evaluation scenarios
python -m goldenhour_rl.data.crisis_generator --output data/crisis_eval.jsonl --num_samples 100 --seed 999
```

### 2. Run Local Verification & Unit Tests
```bash
pytest tests/ -v
```

### 3. Run Baseline Telemetry Benchmark
```bash
python -m goldenhour_rl.eval.benchmark --eval_data data/crisis_eval.jsonl --dry_run
```

### 4. Local CPU Dry-Run of Training Pipeline
```bash
python -m goldenhour_rl.training.train_grpo --data_path data/crisis_eval.jsonl --mock_cpu
```

### 5. Launch Interactive Telemetry Dashboard
```bash
pip install -e ".[demo]"
python spaces/app.py
```
*Visualizes side-by-side triage decisions, ICU/ventilator capacity counters, and Golden-Hour survival margins.*

---

## 6. Cloud GPU Training (Kaggle / Colab)

To train on a free 16GB T4 GPU:

1. Open a new Kaggle Notebook or Google Colab with GPU acceleration enabled.
2. Clone the repository and install requirements:
   ```bash
   !git clone https://github.com/akshaykumarhudedmani/goldenhour-rl.git
   %cd goldenhour-rl
   !pip install -e . unsloth
   ```
3. Run the GRPO training loop:
   ```bash
   !python -m goldenhour_rl.training.train_grpo --max_steps 150 --use_wandb
   ```
4. The trained LoRA adapter will be saved to `outputs/goldenhour_grpo/final_lora`.

---

## 7. Repository Structure

```
goldenhour-rl/
├── .gitignore                      # Enforces zero secret leakage
├── .env.example                    # Environment template
├── pyproject.toml                  # Standards-compliant package definition
├── requirements.txt                # Pinned dependencies
├── README.md                       # Documentation & benchmarks
├── assets/                         # Empirical W&B telemetry charts
│   ├── image_f16a0d.png            # Multiprocessing sandbox vs generation latency
│   ├── image_f17d71.png            # Verifier reward mean & spread doubling
│   ├── image_f17dac.png            # Zero dead batches & mean terminated length
│   └── image_f1806f.png            # Computational parsimony & token compression
├── goldenhour_rl/
│   ├── data/
│   │   └── crisis_generator.py     # Guaranteed-feasible emergency state synthesizer
│   ├── sandbox/
│   │   ├── executor.py             # Multiprocessing timeout sandbox & AST security
│   │   └── verifier.py             # Physical life-support invariant validator
│   ├── training/
│   │   ├── rewards.py              # Parsimony, formatting & physical invariant rewards
│   │   └── train_grpo.py           # Unsloth 4-bit QLoRA + TRL GRPOTrainer harness
│   └── eval/
│       └── benchmark.py            # Out-of-distribution evaluation telemetry
├── spaces/
│   ├── app.py                      # Interactive Gradio Telemetry Benchmark UI
│   └── requirements.txt            # Hugging Face Spaces dependencies
├── scripts/
│   ├── package_release.py          # Builds clean release zip excluding secrets
│   └── generate_telemetry_charts.py # Generates 300-DPI publication charts
├── tests/
│   ├── test_generator.py           # Mathematical feasibility invariant tests
│   └── test_sandbox.py             # AST security, timeouts & invariant unit tests
└── data/                           # Generated synthetic telemetry datasets
```

---

## 8. License

Released under the [MIT License](LICENSE).
