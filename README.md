# GoldenHour-RL: Empirical Mechanics & Failure Modes of SLM Post-Training

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![RL: GRPO](https://img.shields.io/badge/RL-GRPO%20(TRL%20%2B%20Unsloth)-red.svg)](https://github.com/unslothai/unsloth)
[![Model Checkpoint](https://img.shields.io/badge/HF%20Hub-akshaykumarhudedmani%2Fgoldenhour--rl-yellow.svg)](https://huggingface.co/akshaykumarhudedmani/goldenhour-rl)
[![W&B Telemetry](https://img.shields.io/badge/W%26B-goldenhour--rl-orange.svg)](https://wandb.ai/akshaykumarhudedmani-rl/goldenhour-rl)
[![Peak VRAM](https://img.shields.io/badge/Peak%20VRAM-8.11%20GB%20(Train%20G=8)%20%7C%20%3C2.1%20GB%20(Inference)-success.svg)]()
[![Test Suite](https://img.shields.io/badge/Tests-23%2F23%20Passing%20(0.19s)-brightgreen.svg)]()
[![Cost: $0.00](https://img.shields.io/badge/Cost-%240.00%20Zero--Cost-brightgreen.svg)]()

> **"GoldenHour-RL is a testbed for evaluating whether sub-3B Small Language Models (`Qwen/Qwen2.5-Coder-1.5B-Instruct`) can reliably synthesize multi-constraint dispatch logic via verifiable GRPO in resource-constrained environments ($\le 8\text{ GB}$ VRAM)."**

---

## 1. Problem Statement & Research Framing

GoldenHour-RL is a testbed for evaluating whether sub-3B Small Language Models (Qwen2.5-Coder-1.5B) can reliably synthesize multi-constraint dispatch logic via verifiable GRPO in resource-constrained environments ($\le 8\text{ GB}$ VRAM).

Rather than claiming to solve real-world emergency disaster response with a toy setup, this project investigates the systems engineering mechanics, optimization dynamics, and failure modes when post-training small code models on synthetic combinatorial problems.

### The Synthetic Dispatch Benchmark
Each scenario synthesizes a micro-scale crisis dispatch challenge ($3–5$ patients, $2–3$ ambulances, $2–3$ hospitals with constrained ICU beds and ventilators):

$$\max_{\mathbf{x}} \sum_{i \in \mathcal{P}} \mathbb{I}(\text{ETA}_i \le T_i^{\text{SLA}}) \quad \text{s.t.} \quad \sum_{i \in \mathcal{P}} x_{ih} \le C_h^{\text{ICU}}, \quad \sum_{i \in \mathcal{P}, \text{vent}_i=1} x_{ih} \le C_h^{\text{Vent}}, \quad \sum_{h \in \mathcal{H}} x_{iah} \le 1$$

To achieve a valid schedule, the synthesized Python program `get_dispatch_plan()` must satisfy four hard invariants simultaneously:
1. **Ambulance Non-Concurrency:** Each ambulance can transport at most one patient per cycle.
2. **ICU Bed Ceilings:** Hospital ICU bed capacities cannot be exceeded.
3. **Ventilator Availability:** Critical patients requiring mechanical ventilation must be allocated to hospitals with active ventilator capacity.
4. **Golden-Hour Survival SLA:** Transit ETA must strictly fall within each patient's survival deadline ($T_i^{\text{SLA}}$).

### Systems Research Questions
* **Heuristic Failure Modes:** Naive nearest-hospital dispatch heuristics incur catastrophic bottleneck overflows (**66 ICU bed overflows and 47 ventilator overflows** across 50 holdout scenarios), routing all patients into the closest facility regardless of capacity.
* **Pre-Trained SLM Limits:** Zero-shot 1.5B models fail basic execution (14.0% executable), hallucinate hospital entity keys, or enter unclosed syntax loops.
* **Post-Training Dynamics:** Can RL post-training teach a 1.5B parameter model to respect discrete physical invariants using sub-millisecond execution verification? What are the empirical failure modes—such as reward hacking, sandbox thread leakage, and premature policy satiation—encountered along the way?

---

## 2. Systems Architecture & Memory Economics

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
                           | Deterministic Sandbox Verifier (<1ms)|
                           | - AST Loop Bounding (5,000 ceiling) |
                           | - Dynamic Reflection Protection     |
                           | - Zero Ambulance Double-Booking     |
                           | - ICU Bed & Ventilator Limits       |
                           | - Golden-Hour SLA Transit Window    |
                           +-------------------------------------+
                                              |
                        +---------------------+---------------------+
                        |                                           |
             [100% Invariant Compliant]                    [Constraint Violated]
                        |                                           |
                        v                                           v
           Reward: +2.0 + 0.5 * SLA_Margin               Reward: -0.35 to -1.5 (Shaped)
```

### Why GRPO over Actor-Critic PPO?
* **Elimination of the Critic Network:** Traditional Actor-Critic PPO requires holding a separate value model in GPU VRAM alongside the reference and actor models. DeepSeek-style GRPO computes advantages relative to group completions ($G=8$):
  $$A_i = \frac{R_i - \text{mean}(\mathbf{R})}{\text{std}(\mathbf{R}) + \epsilon}$$
* **Memory Economics:** Eliminating the value network enables post-training entirely within consumer or free-tier GPU constraints:
  - **8.11 GB peak VRAM** during $G=8$ rollout generation and backpropagation with Unsloth 4-bit QLoRA on a single Tesla T4 GPU.
  - **$<2.1\text{ GB}$ VRAM** during inference.
* **Deterministic Verification:** Replaces noisy, non-deterministic LLM-as-a-judge evaluators with mathematical invariant checks running in $<0.72\text{ ms}$ per candidate.

---

## 3. Empirical Benchmark Scorecard

Evaluated across $N=50$ out-of-distribution crisis scenarios featuring adversarial trap routes and bottlenecked hospital capacities:

| Strategy | Engine Type | Executable Plan % | Zero-Violation Rate | Strict Survival % | ICU Bed Overflows | Ventilator Overflows | Total Violations |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Greedy Nearest Hospital** | Programmatic Heuristic Baseline | 100.0% | 34.0% | 70.6% | 66 | 47 | 119 |
| **Base Qwen-1.5B-Instruct** | Zero-Shot Pre-Trained LLM (Pass@1, $T=0.2$) | 14.0% | 0.0% | 2.0% | 2 | 2 | 205 |
| **Base Qwen-1.5B-Instruct** | Zero-Shot Pre-Trained LLM (Best-of-4, $T=0.7$) | 84.0% | 12.0% | 43.8% | 7 | 17 | 120 |
| **GoldenHour-RL v0.1** | Verifier-Guided LoRA (Best-of-4, $T=0.7$) | 86.0% | 8.0% | 37.8% | 8 | 8 | 155 |
| **GoldenHour-RL v0.2** | Curriculum-Hardened GRPO (Best-of-4, $T=0.7$) | 86.0% | 8.0% | 41.3% | 11 | 17 | 133 |
| **GoldenHour-RL v0.2** | Policy Sampling (Pass@1, $T=0.2$) | 8.0% | 0.0% | 1.0% | 2 | 0 | 213 |
| **GoldenHour-RL v0.3 (SFT + GRPO)** | Stage 2 LoRA (Pass@1, $T=0.2$) | **94.0%** | **78.0%** | **88.2%** | **4** | **0** | **16** |
| **GoldenHour-RL v0.3 (SFT + GRPO)** | Stage 2 LoRA (Best-of-4, $T=0.7$) | **100.0%** | **92.0%** | **96.5%** | **1** | **0** | **5** |
| **Oracle Target Schedule** | Ground-Truth Solver (Mathematical Upper Bound) | 100.0% | **100.0%** | **100.0%** | **0** | **0** | **0** |

### Benchmark Analysis
1. **The Role of Stage 1 SFT Anchoring:** Anchoring control flow syntax through 300 SFT examples prior to GRPO broke through the cold-start exploration void, moving zero-violation rates from 8.0% in v0.2 up to **78.0% (Pass@1)** and **92.0% (Best-of-4)**.
2. **Ventilator Constraint Adherence:** In both Pass@1 and Best-of-4 modes, GoldenHour-RL v0.3 incurred **0 ventilator overflows**, confirming the model internalized hard equipment constraints.
3. **Inference-Time Verifier Rejection (Best-of-4):** Executable plan generation reached **100.0% under Verifier Best-of-4**, leveraging the $<1\text{ ms}$ sandbox to filter out non-viable candidate programs before dispatch.
4. **Heuristic Failure:** While the greedy baseline reports high nominal patient dispatch, it incurs **119 critical physical violations** (66 ICU overflows, 47 ventilator overflows) because it lacks multi-facility lookahead.

---

## 4. Empirical Training Telemetry & Systems Profiling

Telemetry recorded from the production 50-step two-stage curriculum run (`checkpoint-25` and `checkpoint-50`, weights mirrored at [`akshaykumarhudedmani/goldenhour-rl`](https://huggingface.co/akshaykumarhudedmani/goldenhour-rl)) on a single Tesla T4 GPU ($5.34\text{ hours}$ wall-clock, $8.11\text{ GB}$ peak VRAM during $G=8$ rollouts, $<2.1\text{ GB}$ during inference):

| Step | Loss | Format Reward | Verifier Reward | KL Divergence | Mean Completion Length | Operational Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **5** | `-0.1656` | `0.909` | `+1.655` | `0.000` | 495.1 tokens | Syntax anchored via SFT primer; positive gradient emergence |
| **10** | `-0.0993` | `0.928` | `+1.746` | `0.000` | 537.4 tokens | Feasible priority queues formed; zero syntax drift |
| **15** | `-0.0449` | `0.966` | `+2.148` | `0.000` | 568.4 tokens | Ascending into zero-violation jackpot zone ($>+2.0$) |
| **20** | `-0.0102` | `0.969` | `+2.220` | `0.000` | 576.7 tokens | Hard ventilator and ICU capacity constraints strictly observed |
| **25** | `-0.0211` | `0.950` | `+2.155` | `0.000` | 581.4 tokens | Convergence Check: Format $\ge 0.8$, Verifier $\ge +2.0$ (Unconstrained objective with $\beta=0.0$) |
| **30** | `-0.0284` | `0.981` | `+2.196` | `0.000` | 576.3 tokens | Consistent constraint satisfaction across adversarial topologies |
| **35** | `-0.0176` | `0.961` | `+2.156` | `0.000` | 579.1 tokens | Zero OOMs; completion length stabilized |
| **40** | `-0.0192` | `0.981` | `+2.235` | `0.000` | 579.9 tokens | Peak reward plateau; exploration variance beginning to narrow |
| **45** | `+0.0008` | `0.956` | `+2.234` | `0.000` | 582.6 tokens | Gradient norms declining toward zero |
| **50** | `-0.0000` | `0.994` | `+2.228` | `0.000` | 584.0 tokens | Policy convergence plateaus at 99.4% syntax and +2.23 verifier reward; onset of policy satiation (see Ablation 4) |

### Optimization Dynamics & Telemetry Notes
* **Note on KL Divergence & DAPO ($\beta = 0.0$):** We trained using an unconstrained GRPO / Direct Advantage Policy Optimization (DAPO) variant with reference model KL penalty set to zero ($\beta = 0.0$). Consequently, the logged KL divergence is identically $0.000$ across all steps by design, rather than an active constraint or milestone.
* **Stage 1 SFT Anchoring Eliminates Cold-Start Drift:** Unsloth 4-bit QLoRA initialization anchored code formatting early, starting format rewards at $0.909$ at Step 5 and reaching $0.994$ by Step 50.
* **Natural Token Termination:** Completions stabilized at $\sim 580$ tokens with natural `<|im_end|>` termination, preventing runaway loops or verbose reasoning bloat without aggressive manual clipping.
* **Sub-Millisecond Verification:** The compiled AST sandbox verified candidate dispatch plans in **$<0.72\text{ ms}$ per candidate**, allowing multi-candidate group rollouts ($G=8$) to execute without creating a CPU-side training bottleneck.

---

## 5. Forensic Systems Ablations & Post-Mortems

The primary engineering insights of GoldenHour-RL emerged from diagnosing empirical failure modes:

### Ablation 1: The Specification-Gaming Reward Trap
* **Symptom:** In early v0.1 training (`lja961ve`), the model achieved 0 capacity overflows, yet evaluation showed 0.5% patient survival.
* **Root Cause Analysis:** The format reward awarded $+0.5$ whenever code matched `def get_dispatch_plan():`. Because assigning zero patients trivially causes 0 ICU and 0 ventilator overflows, the model discovered that emitting `return []` avoided all capacity penalties while harvesting format bonuses.
* **The Fix:**
  1. **Execution-Gated Format Scoring:** Implemented an execution gate in `goldenhour_rl/training/rewards.py` ensuring that crashing code or empty return lists (`return []`) immediately zero out or penalize format rewards ($R_{\text{format}} \le 0.0$).
  2. **Monotonic Penalty Asymmetry:** Heavily penalized unassigned patients ($-0.50$) relative to capacity overflows ($-0.35$), ensuring that leaving patients behind is strictly worse than attempting an assignment:
     ```python
     # rewards.py: Monotonic Penalty Asymmetry
     unassigned_penalty = len(unassigned) * -0.50
     overflow_penalty = (icu_overflows + vent_overflows) * -0.35
     ```

### Ablation 2: Sandbox Isolation & The CPython GIL Deadlock
* **Symptom:** Early evaluation threads hung indefinitely on malformed model completions containing nested `while True` loops.
* **Root Cause Analysis:** Standard Python multi-threading cannot terminate running CPython bytecode asynchronously without GIL deadlocks or thread pool contamination.
* **The Fix:**
  * Implemented `LoopBoundingTransformer(ast.NodeTransformer)` in `goldenhour_rl/sandbox/executor.py`.
  * Before execution, the AST is statically rewritten to inject an iteration counter capped at 5,000 iterations for every `For` and `While` loop:
    ```python
    # Compiles and terminates infinite loops in <0.15ms
    class LoopBoundingTransformer(ast.NodeTransformer):
        def visit_While(self, node):
            # Injects: __loop_guard_counter += 1; if __loop_guard_counter > 5000: raise TimeoutError
    ```

### Ablation 3: Schema Drift & Dynamic Reflection Security
* **Symptom:** Generated solutions occasionally raised `AttributeError: 'list' object has no attribute 'values'` or failed security audits via `getattr(getattr((), "__ca" + "lass__"), "__ba" + "ses__")[0]`.
* **The Fix:**
  1. **`DualHospitalMap`:** Subclassed `dict` to support both integer index access (`hospitals[0]`) and dictionary key access (`hospitals["HOSP_1"]`), alongside `.values()`.
  2. **Dynamic Reflection Defenses:** Wrapped reflection builtins with `_safe_getattr` and `_safe_hasattr`, blocking any attribute string starting with `_` to eliminate dunder escape vulnerabilities (`__subclasses__`, `__globals__`).
  3. **Safe Builtins:** Whitelisted `__build_class__`, `object`, `type`, `super`, `property`, `next`, `iter`, alongside boolean aliases (`true: True`, `false: False`, `null: None`).

### Ablation 4: Policy Satiation & SFT Template Memorization
* **Symptom:** By Step 45–50, `frac_reward_zero_std` reached $0.95$ and gradient norms declined toward $0.0$, despite high nominal verifier rewards ($+2.23$).
* **Root Cause:** Pre-training on 300 SFT algorithmic demonstrations caused the 1.5B model to internalize the canonical greedy priority queue. During GRPO rollouts, all 8 completions in a group generated near-identical code, eliminating advantage variance ($\sigma_R \to 0$).
* **Finding:** Demonstrates that for deterministic algorithmic tasks, cold-start SFT easily causes premature policy saturation in small models unless paired with high rollout temperature ($T \ge 0.8$) or dynamic syntactic perturbations.

---

## 6. Repository Layout

```
goldenhour-rl/
├── data/
│   ├── crisis_train.jsonl         # 300 synthetic training scenarios with adversarial trap routes
│   └── crisis_eval.jsonl          # 50 out-of-distribution evaluation scenarios (seed=999)
├── goldenhour_rl/
│   ├── __init__.py                # Package version export (__version__ = "0.3.0")
│   ├── data/
│   │   ├── __init__.py
│   │   └── crisis_generator.py    # Procedural emergency telemetry & trap-route generator
│   ├── sandbox/
│   │   ├── __init__.py
│   │   ├── executor.py            # AST loop-bounding sandbox, safe builtins & namespace injection
│   │   └── verifier.py            # Physical invariant checker (ICU, vents, single ambulance, SLA)
│   ├── training/
│   │   ├── __init__.py
│   │   ├── rewards.py             # Anti-gaming reward functions (execution-gated format scoring)
│   │   └── train_grpo.py          # Unsloth 4-bit QLoRA + TRL GRPOTrainer harness
│   └── eval/
│       ├── __init__.py
│       └── benchmark.py           # Dual-mode benchmark (Pass@1 vs. Verifier-Guided Best-of-4)
├── kaggle_kernel/
│   ├── kernel-metadata.json       # Kaggle API metadata
│   ├── train_kaggle.py            # Headless 50-step training driver with automatic HF Hub push
│   └── eval_kaggle.py             # Headless benchmark evaluation runner
├── scripts/
│   ├── generate_telemetry_charts.py # Generates 300-DPI publication plots from W&B run history
│   ├── package_release.py         # Deterministic release packager (strips secrets, caches, logs)
│   ├── poll_wandb.py              # Telemetry monitor script
│   └── check_wandb.py             # W&B health-check utility
├── spaces/
│   ├── app.py                     # Hugging Face Space Gradio interactive triage visualizer
│   └── requirements.txt           # Lightweight dependencies for Space deployment
├── tests/
│   ├── __init__.py
│   ├── test_generator.py          # Feasibility, zero-leakage, and trap-route unit tests
│   └── test_sandbox.py            # AST security, infinite-loop bounding, and reward-surface tests
├── .env.example                   # Secret template (HF_TOKEN, WANDB_API_KEY placeholders)
├── .gitignore                     # Strict exclusion of .env, caches, local weights, and zips
├── LICENSE                        # MIT License
├── pyproject.toml                 # Modern package build configuration (setuptools)
├── requirements.txt               # Pinned runtime dependencies
└── README.md                      # Comprehensive technical architecture & ablation report
```

---

## 7. Limitations & Research Handoff

1. **Sub-3B Parameter Capacity:** While Qwen2.5-Coder-1.5B internalizes priority queue dispatch templates, small models exhibit high sensitivity to sampling temperature and lack broader combinatorial reasoning outside structured prompt scaffolds.
2. **The Neuro-Symbolic Boundary:** For production dispatch logistics, language models should not perform raw branch-and-bound optimization. A practical architecture employs the LLM strictly to parse unstructured natural language telemetry into structured constraints, handing off the constraint matrix to a deterministic Mixed-Integer Linear Programming (MILP) solver (e.g., Google OR-Tools / HiGHS).
3. **Rollout Diversity in GRPO:** As shown in Ablation 4, cold-start SFT combined with low sampling temperature leads to advantage collapse. Addressing policy satiation requires dynamic temperature scaling, syntactic mutation of prompt states, or entropy regularization during group rollout generation.

---

## 8. Quickstart & Reproducibility Guide

### Installation
```bash
git clone https://github.com/akshaykumarhudedmani/goldenhour-rl.git
cd goldenhour-rl
pip install -e ".[dev]"
```

### Environment Configuration
Copy `.env.example` to `.env` and provide your credentials:
```bash
cp .env.example .env
# Edit .env: HF_TOKEN=... WANDB_API_KEY=...
```

### Run Unit Test Suite (<0.20s AST Verification)
```bash
pytest -v
```
Output:
```
============================= 23 passed in 0.19s ==============================
```

### Synthesize Emergency Scenarios
```bash
# Generate 300 training scenarios
python -m goldenhour_rl.data.crisis_generator --output data/crisis_train.jsonl --num_samples 300 --seed 42

# Generate 50 out-of-distribution evaluation scenarios
python -m goldenhour_rl.data.crisis_generator --output data/crisis_eval.jsonl --num_samples 50 --seed 999
```

### Run Baseline Heuristic Benchmark
```bash
python -m goldenhour_rl.eval.benchmark --eval_data data/crisis_eval.jsonl --dry_run
```

### Local CPU Pipeline Dry-Run
```bash
python -m goldenhour_rl.training.train_grpo --data_path data/crisis_eval.jsonl --mock_cpu
```

### Launch Interactive Gradio Telemetry Space
```bash
pip install -e ".[demo]"
python spaces/app.py
```

---

## 9. License

This project is licensed under the [MIT License](LICENSE).
