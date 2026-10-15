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
* **Eliminates the Critic Network:** Traditional Actor-Critic PPO requires holding a separate value model in GPU VRAM. GRPO computes advantages relative to group completions ($G=6$), cutting training VRAM by **~45%** and enabling training on a single free 16GB T4 GPU.
* **Deterministic Life-Support Invariants:** Replaces noisy LLM-as-a-judge with mathematical ground-truth verification.

---

## 2. Benchmark Telemetry: Heuristic vs. Verifiable Reference

Evaluated across $N=100$ out-of-distribution crisis scenarios:

| Strategy | Zero-Violation Rate | Patient Survival (SLA) | ICU Bed Overflows | Ventilator Overflows | Total Violations |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Greedy Nearest Hospital** | 37.0% | 91.3% | 62 | 13 | 109 |
| **GoldenHour-RL Reference** | **100.0%** | **100.0%** | **0** | **0** | **0** |

---

## 3. The $0.00 Extra-Cost Architecture

| Resource Tier | Cost | Hardware Target | Purpose |
| :--- | :---: | :--- | :--- |
| **Tier 1: Local Machine** | **$0.00** | Any CPU / 4GB+ Laptop | Scenario synthesis, deterministic sandbox verifier, and unit tests via `pytest`. |
| **Tier 2: Free Cloud GPU** | **$0.00** | Kaggle (2x T4 32GB) or Google Colab (T4 16GB) | 2-hour Unsloth 4-bit QLoRA + GRPO training loop (Peak VRAM: ~8.4 GB). |
| **Tier 3: HF Space** | **$0.00** | Hugging Face Spaces | Live telemetry visualization dashboard. |

---

## 4. Quickstart & Local Setup

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

---

## 5. Cloud GPU Training (Kaggle / Colab)

To train on a free 16GB T4 GPU:

1. Open a new Kaggle Notebook or Google Colab with GPU acceleration enabled.
2. Clone the repository and install requirements:
   ```bash
   !git clone https://github.com/akshaykumarhudedmani/goldenhour-rl.git
   %cd goldenhour-rl
   !pip install -r requirements.txt unsloth
   ```
3. Run the GRPO training loop:
   ```bash
   !python -m goldenhour_rl.training.train_grpo --max_steps 150 --use_wandb
   ```
4. The trained LoRA adapter will be saved to `outputs/goldenhour_grpo/final_lora`.

---

## 6. Repository Structure

```
goldenhour-rl/
├── .gitignore                      # Enforces zero secret leakage
├── .env.example                    # Environment template
├── requirements.txt                # Pinned dependencies
├── README.md                       # Documentation & benchmarks
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
├── tests/
│   ├── test_generator.py           # Mathematical feasibility invariant tests
│   └── test_sandbox.py             # AST security, timeouts & invariant unit tests
└── data/                           # Generated synthetic telemetry datasets
```

---

## 7. License

Released under the [MIT License](LICENSE).
