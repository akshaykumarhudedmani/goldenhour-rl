import json
from goldenhour_rl.data.crisis_generator import generate_scenario
from goldenhour_rl.sandbox.executor import execute_dispatch_code, validate_ast
from goldenhour_rl.sandbox.verifier import verify_dispatch_plan
from goldenhour_rl.training.rewards import format_reward_func, verifier_reward_func


def test_infinite_loop_killed():
    """Verify that execution sandbox terminates infinite loops within timeout without hanging."""
    malicious_code = "def get_dispatch_plan():\n    while True:\n        pass\n"
    res = execute_dispatch_code(malicious_code, timeout_sec=0.5)
    assert res["ok"] is False
    assert res["error"] == "ExecutionTimeout"


def test_ast_security_catches_imports_and_banned_functions():
    """Verify AST security checks catch dangerous imports and unsafe builtins."""
    import_code = "import os\ndef get_dispatch_plan():\n    return []\n"
    assert "SecurityViolation" in validate_ast(import_code)

    eval_code = "def get_dispatch_plan():\n    eval('2+2')\n    return []\n"
    assert "SecurityViolation" in validate_ast(eval_code)


def test_reference_solution_is_valid():
    """Verify that synthetic scenarios always include a physically valid reference solution."""
    sample = generate_scenario(seed=777)
    ref_plan = json.loads(sample["reference_solution"])
    ver_res = verify_dispatch_plan(ref_plan, sample["state_json"])
    assert ver_res["valid"] is True
    assert len(ver_res["violations"]) == 0
    assert ver_res["sla_score"] >= 0.0


def test_hallucinated_hospital_caught():
    sample = generate_scenario(seed=101)
    state = json.loads(sample["state_json"])
    bad_plan = [
        {"patient_id": p["patient_id"], "ambulance_id": "AMB_1", "hospital_id": "HOSP_GHOST"}
        for p in state["patients"]
    ]
    res = verify_dispatch_plan(bad_plan, sample["state_json"])
    assert res["valid"] is False
    assert any("Hallucination" in v for v in res["violations"])


def test_double_booked_ambulance_caught():
    sample = generate_scenario(seed=102)
    state = json.loads(sample["state_json"])
    # Assign the exact same ambulance to all patients
    amb_id = state["ambulances"][0]["ambulance_id"]
    hosp_id = list(state["hospitals"].keys())[0]
    bad_plan = [
        {"patient_id": p["patient_id"], "ambulance_id": amb_id, "hospital_id": hosp_id}
        for p in state["patients"]
    ]
    res = verify_dispatch_plan(bad_plan, sample["state_json"])
    assert res["valid"] is False
    assert any("DoubleBooking" in v for v in res["violations"])


def test_nested_dict_ids_handled_safely():
    """Verify that nested dicts and unhashable structures in plan IDs are sanitized without TypeError."""
    sample = generate_scenario(seed=103)
    state = json.loads(sample["state_json"])
    # Emit ambulance_id and hospital_id as nested dicts, and patient_id as dict
    bad_plan = [
        {
            "patient_id": {"patient_id": p["patient_id"]},
            "ambulance_id": {"ambulance_id": state["ambulances"][0]["ambulance_id"]},
            "hospital_id": {"id": list(state["hospitals"].keys())[0]},
        }
        for p in state["patients"]
    ]
    # This must not crash with TypeError: unhashable type: 'dict'
    res = verify_dispatch_plan(bad_plan, sample["state_json"])
    assert isinstance(res, dict)
    assert "violations" in res



def test_reward_pipeline_end_to_end():
    sample = generate_scenario(seed=202)
    valid_plan_json = sample["reference_solution"]

    mock_valid_completion = (
        "<think>Analyzing patient survival windows and ventilator availability across hospitals.</think>\n"
        f"```python\n"
        f"def get_dispatch_plan():\n"
        f"    return {valid_plan_json}\n"
        f"```"
    )

    # Format reward check
    fmt_scores = format_reward_func([sample["prompt"]], [mock_valid_completion])
    assert fmt_scores[0] > 0.0

    # Verifier reward check (should give high positive reward)
    ver_scores = verifier_reward_func(
        [sample["prompt"]],
        [mock_valid_completion],
        state_json=[sample["state_json"]]
    )
    assert ver_scores[0] >= 2.0


def test_malformed_completion_penalized():
    sample = generate_scenario(seed=303)
    mock_bad_completion = "I would assign the first ambulance to patient 1."

    fmt_scores = format_reward_func([sample["prompt"]], [mock_bad_completion])
    assert fmt_scores[0] < 0.0

    ver_scores = verifier_reward_func(
        [sample["prompt"]],
        [mock_bad_completion],
        state_json=[sample["state_json"]]
    )
    assert ver_scores[0] < 0.0


def test_virtual_file_and_input_execution():
    """Verify that generated code using import json and open('telemetry.json') executes safely."""
    sample = generate_scenario(seed=404)
    code = (
        "import json\n"
        "def get_dispatch_plan():\n"
        "    with open('emergency_telemetry.json', 'r') as f:\n"
        "        data = json.load(f)\n"
        "    return []\n"
    )
    res = execute_dispatch_code(code, timeout_sec=2.0, state_str=sample["state_json"])
    assert res["ok"] is True
    assert res["plan"] == []


def test_global_data_and_parameterized_dispatch_plan():
    """Verify that code using global data or get_dispatch_plan(data) executes safely."""
    sample = generate_scenario(seed=505)
    # 1. Global data access
    code_global = (
        "def get_dispatch_plan():\n"
        "    p = data['patients']\n"
        "    return []\n"
    )
    res1 = execute_dispatch_code(code_global, timeout_sec=2.0, state_str=sample["state_json"])
    assert res1["ok"] is True
    assert res1["plan"] == []

    # 2. Parameterized get_dispatch_plan(state)
    code_param = (
        "def get_dispatch_plan(telemetry):\n"
        "    p = telemetry['patients']\n"
        "    return []\n"
    )
    res2 = execute_dispatch_code(code_param, timeout_sec=2.0, state_str=sample["state_json"])
    assert res2["ok"] is True
    assert res2["plan"] == []


def test_reward_hacking_eliminated():
    """Verify that crashing code is strictly worse than partial plans, eliminating the reward hack."""
    sample = generate_scenario(seed=808)

    # 1. Crashing code
    crash_comp = "<think>Plan</think>\n```python\ndef get_dispatch_plan():\n    raise ValueError('crash')\n```"
    ver_crash = verifier_reward_func([sample["prompt"]], [crash_comp], state_json=[sample["state_json"]])[0]

    # 2. Missing get_dispatch_plan
    hollow_comp = "<think>Plan</think>\n```python\nimport json\nx = 1\n```"
    ver_hollow = verifier_reward_func([sample["prompt"]], [hollow_comp], state_json=[sample["state_json"]])[0]

    # 3. Empty return [] plan
    empty_comp = "<think>Plan</think>\n```python\ndef get_dispatch_plan():\n    return []\n```"
    ver_empty = verifier_reward_func([sample["prompt"]], [empty_comp], state_json=[sample["state_json"]])[0]
    fmt_empty = format_reward_func([sample["prompt"]], [empty_comp])[0]

    # 4. Valid reference solution
    ref_plan = json.loads(sample["reference_solution"])
    valid_comp = f"<think>Plan</think>\n```python\ndef get_dispatch_plan(state):\n    return {json.dumps(ref_plan)}\n```"
    ver_valid = verifier_reward_func([sample["prompt"]], [valid_comp], state_json=[sample["state_json"]])[0]

    # Assert strictly monotonic gradient: crashing (-1.5) == hollow (-1.5) < empty plan (-1.0) < valid (>= 2.0)
    assert ver_crash == -1.5
    assert ver_hollow == -1.5
    assert ver_empty == -1.0
    assert fmt_empty < 0.0
    assert ver_valid >= 2.0


def test_typing_and_safe_builtins_allowed():
    """Verify that models importing typing and using safe functional builtins pass AST validation and execute."""
    typed_code = (
        "from typing import List, Dict, Optional\n"
        "import heapq, collections, itertools\n"
        "\n"
        "def get_dispatch_plan(state: Optional[Dict] = None) -> List[Dict]:\n"
        "    if state is None or not any(state.get('patients', [])):\n"
        "        return []\n"
        "    ans = [p for p in state['patients'] if isinstance(p, dict)]\n"
        "    return []\n"
    )
    sample = generate_scenario(seed=909)
    res = execute_dispatch_code(typed_code, timeout_sec=2.0, state_str=sample["state_json"])
    assert res["ok"] is True
    assert res["plan"] == []


def test_raw_list_and_variable_fallback():
    """Verify that raw list literals and dispatch_plan variable assignments execute cleanly."""
    sample = generate_scenario(seed=1010)
    ref_plan = json.loads(sample["reference_solution"])

    # Raw list literal
    raw_list_code = json.dumps(ref_plan)
    res_list = execute_dispatch_code(raw_list_code, timeout_sec=2.0, state_str=sample["state_json"])
    assert res_list["ok"] is True
    assert len(res_list["plan"]) == len(ref_plan)

    # Variable assignment without def get_dispatch_plan
    var_code = f"dispatch_plan = {json.dumps(ref_plan)}\n"
    res_var = execute_dispatch_code(var_code, timeout_sec=2.0, state_str=sample["state_json"])
    assert res_var["ok"] is True
    assert len(res_var["plan"]) == len(ref_plan)


def test_oop_class_and_json_literals_execution():
    """Verify that models defining OOP helper classes and using true/false/null execute cleanly."""
    oop_code = (
        "class Patient:\n"
        "    def __init__(self, patient_id, severity, requires_vent):\n"
        "        self.patient_id = patient_id\n"
        "        self.severity = severity\n"
        "        self.requires_vent = requires_vent\n"
        "\n"
        "def get_dispatch_plan():\n"
        "    # Test JSON booleans and idempotent json.loads\n"
        "    t = true\n"
        "    f = false\n"
        "    n = null\n"
        "    # Test loading pre-parsed telemetry_data\n"
        "    parsed = json.loads(telemetry_data)\n"
        "    p = Patient('PAT_1', 1, t)\n"
        "    return []\n"
    )
    sample = generate_scenario(seed=1212)
    res = execute_dispatch_code(oop_code, timeout_sec=2.0, state_str=sample["state_json"])
    assert res["ok"] is True
    assert res["plan"] == []


def test_specification_gaming_positive_reward_trap_eliminated():
    """Verify that empty plans or crashing code cannot earn positive format reward or net positive reward."""
    sample = generate_scenario(seed=1313)

    # Empty plan with filler reasoning trace
    empty_comp = (
        "<think>Triage analysis: evaluating capacity constraints and patient priority carefully.</think>\n"
        "```python\n"
        "def get_dispatch_plan():\n"
        "    return []\n"
        "```"
    )
    fmt_empty = format_reward_func([sample["prompt"]], [empty_comp], state_json=[sample["state_json"]])[0]
    ver_empty = verifier_reward_func([sample["prompt"]], [empty_comp], state_json=[sample["state_json"]])[0]
    assert fmt_empty <= 0.0, f"Format reward for empty plan must be non-positive, got {fmt_empty}"
    assert ver_empty == -1.0
    assert (fmt_empty + ver_empty) <= -1.0

    # Crashing code with filler reasoning trace
    crash_comp = (
        "<think>Triage analysis: evaluating capacity constraints and patient priority carefully.</think>\n"
        "```python\n"
        "def get_dispatch_plan():\n"
        "    raise RuntimeError('crash')\n"
        "```"
    )
    fmt_crash = format_reward_func([sample["prompt"]], [crash_comp], state_json=[sample["state_json"]])[0]
    ver_crash = verifier_reward_func([sample["prompt"]], [crash_comp], state_json=[sample["state_json"]])[0]
    assert fmt_crash <= 0.0, f"Format reward for crashing code must be non-positive, got {fmt_crash}"
    assert ver_crash == -1.5
    assert (fmt_crash + ver_crash) <= -1.5


def test_safe_getattr_and_hasattr_security():
    """Verify that _safe_getattr allows safe attribute lookups but strictly blocks private/dunder traversal."""
    class Dummy:
        def __init__(self):
            self.patient_id = "PAT_1"
            self._secret = "hidden"

    d = Dummy()
    # Benign lookups via execute_dispatch_code
    code_ok = (
        "def get_dispatch_plan():\n"
        "    class Item:\n"
        "        val = 42\n"
        "    x = getattr(Item, 'val', 0)\n"
        "    has_val = hasattr(Item, 'val')\n"
        "    assert x == 42 and has_val is True\n"
        "    return []\n"
    )
    res_ok = execute_dispatch_code(code_ok)
    assert res_ok["ok"] is True

    # Malicious lookup trying to reach dunders dynamically
    code_bad = (
        "def get_dispatch_plan():\n"
        "    dunder = '__cl' + 'ass__'\n"
        "    cls = getattr([], dunder)\n"
        "    return []\n"
    )
    res_bad = execute_dispatch_code(code_bad)
    assert res_bad["ok"] is False
    assert "SecurityViolation" in res_bad.get("error", "") or "PermissionError" in res_bad.get("error", "")





