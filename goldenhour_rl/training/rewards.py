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
    Evaluates format compliance:
    - Enforces reasoning trace inside <think>...</think> tags.
    - Enforces executable python block inside ```python ... ```.
    - Robust to terminal backtick clipping while rewarding properly closed code blocks.
    - Rewards concise, non-bloated reasoning (under 200 words / 800 chars).
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
            score += 0.4
            think_text = think_match.group(1).strip()
            think_len = len(think_text)
            # Parsimony bonus: reward concise reasoning, penalize token bloat
            if 30 <= think_len <= 800:
                score += 0.2
        else:
            score -= 0.6

        # Code block formatting
        if code_match:
            code_text = code_match.group(1).strip()
            # Reward closed code block higher than clipped unclosed block
            if CLOSED_CODE_PATTERN.search(text):
                score += 0.3
            else:
                score += 0.1

            if "def get_dispatch_plan" in code_text:
                score += 0.2
        else:
            score -= 0.6

        rewards.append(round(score, 3))

    return rewards


def verifier_reward_func(prompts: List[Any], completions: List[Any], state_json: List[str] = None, **kwargs) -> List[float]:
    """
    Evaluates physical invariant compliance inside the deterministic sandbox:
    - +2.0 for 100% zero-violation physical dispatch schedules.
    - Up to +0.5 additional bonus for maximizing Golden-Hour arrival margin.
    - Graded penalty shaping for constraint violations (ICU/vent overflow, double-booking, SLA breach).
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
            rewards.append(-1.0)
            continue

        code_str = code_match.group(1).strip()
        if not code_str:
            rewards.append(-1.0)
            continue

        exec_res = execute_dispatch_code(code_str, timeout_sec=2.0, state_str=s_json)

        if not exec_res["ok"]:
            # Penalize syntax error, execution crash, or timeout
            rewards.append(-0.8)
            continue

        ver_res = verify_dispatch_plan(exec_res["plan"], s_json)

        if ver_res["valid"]:
            # +2.0 for 100% physically valid schedule + SLA margin bonus
            rewards.append(round(2.0 + 0.5 * ver_res["sla_score"], 4))
        else:
            # Partial shaping penalty based on number of violations
            penalty = min(1.2, 0.25 * len(ver_res["violations"]))
            rewards.append(round(-0.2 - penalty, 4))

    return rewards
