import re
from typing import List, Any
from goldenhour_rl.sandbox.executor import execute_dispatch_code
from goldenhour_rl.sandbox.verifier import verify_dispatch_plan

THINK_PATTERN = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL)
CODE_PATTERN = re.compile(r"```python\s*(.*?)(?:```|$)", flags=re.DOTALL)
CLOSED_CODE_PATTERN = re.compile(r"```python\s*(.*?)\s*```", flags=re.DOTALL)


def _extract_completion_text(comp: Any) -> str:
    """Extracts raw text from completion whether it's a string or chat list format."""
    if isinstance(comp, list) and len(comp) > 0:
        if isinstance(comp[0], dict) and "content" in comp[0]:
            return comp[0]["content"]
        return str(comp[0])
    elif isinstance(comp, dict) and "content" in comp:
        return comp["content"]
    return str(comp)


def format_reward_func(prompts: List[Any], completions: List[Any], **kwargs) -> List[float]:
    """
    Evaluates format compliance without reward hacking:
    - Enforces reasoning trace inside <think>...</think> tags.
    - Enforces properly formatted python block defining get_dispatch_plan.
    - Penalizes hollow or missing function structures.
    """
    rewards = []
    for comp in completions:
        text = _extract_completion_text(comp)
        think_match = THINK_PATTERN.search(text)
        code_match = CODE_PATTERN.search(text)

        if not think_match and not code_match:
            rewards.append(-1.5)
            continue

        score = 0.0

        # Think tag formatting
        if think_match:
            score += 0.3
            think_text = think_match.group(1).strip()
            think_len = len(think_text)
            if 30 <= think_len <= 800:
                score += 0.2
        else:
            score -= 0.5

        # Code block formatting
        if code_match:
            code_text = code_match.group(1).strip()
            if "def get_dispatch_plan" in code_text:
                if re.search(r"return\s*\[\s*\]", code_text):
                    score -= 0.5  # Penalize empty dispatch list hack
                else:
                    score += 0.3
                    if CLOSED_CODE_PATTERN.search(text):
                        score += 0.2
            else:
                score -= 0.4
        else:
            score -= 0.5

        rewards.append(round(score, 3))

    return rewards


def verifier_reward_func(prompts: List[Any], completions: List[Any], state_json: List[str] = None, **kwargs) -> List[float]:
    """
    Evaluates physical invariant compliance with dense, monotonic reward shaping:
    - -1.5 for code crashes, syntax errors, or unhandled exceptions.
    - -1.0 penalty for empty dispatch plans (return []).
    - Dense progressive credit for each patient successfully arriving on time without capacity overflow.
    - -0.25 penalty per unassigned patient; -0.4 penalty per ICU/ventilator capacity overflow.
    - +2.0 jackpot bonus for 100% zero-violation dispatch plans.
    """
    if state_json is None:
        state_json = kwargs.get("state_json", [None] * len(completions))

    rewards = []
    for comp, s_json in zip(completions, state_json):
        if not s_json:
            rewards.append(0.0)
            continue

        text = _extract_completion_text(comp)
        code_match = CODE_PATTERN.search(text)

        if not code_match:
            rewards.append(-1.5)
            continue

        code_str = code_match.group(1).strip()
        if not code_str or "def get_dispatch_plan" not in code_str:
            rewards.append(-1.5)
            continue

        exec_res = execute_dispatch_code(code_str, timeout_sec=2.0, state_str=s_json)

        if not exec_res["ok"]:
            # Harsh penalty for execution crash, syntax error, or timeout
            rewards.append(-1.5)
            continue

        plan = exec_res.get("plan")
        if not isinstance(plan, list) or len(plan) == 0:
            # Penalize empty dispatch plan (leaving all patients on the street)
            rewards.append(-1.0)
            continue

        ver_res = verify_dispatch_plan(plan, s_json)

        if ver_res["valid"]:
            # Perfect schedule: +2.0 jackpot + SLA arrival margin bonus
            rewards.append(round(2.0 + 0.5 * ver_res["sla_score"], 4))
        else:
            # Dense progressive shaping: reward successful patient routings, penalize unassigned/overflows
            stats = ver_res.get("stats", {})
            total_p = max(1, stats.get("total_patients", 1))
            survived_p = stats.get("survived_count", 0)
            unassigned = stats.get("unassigned_count", 0)
            icu_ov = stats.get("icu_overflow_count", 0)
            vent_ov = stats.get("vent_overflow_count", 0)

            patient_ratio = survived_p / total_p  # 0.0 to 1.0

            # Strictly monotonic gradient: -0.2 base invalid, credit for survived, penalties for bottlenecks
            score = -0.2 + (1.2 * patient_ratio) - (0.25 * unassigned) - (0.4 * icu_ov) - (0.4 * vent_ov)
            clamped_score = max(-1.0, min(1.0, score))
            rewards.append(round(clamped_score, 4))

    return rewards
