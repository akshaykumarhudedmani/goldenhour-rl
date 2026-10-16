import ast
import io
import json
import math
import collections
import inspect
import itertools
import multiprocessing
import threading
from typing import Dict, Any, Optional

# Safe modules allowed to be imported in generated dispatch functions
SAFE_MODULES = {
    "json", "math", "re", "collections", "itertools", "heapq", "typing", "copy", "dataclasses", "enum"
}

# Safe builtins allowed in generated dispatch functions
SAFE_BUILTINS = {
    "dict": dict,
    "list": list,
    "set": set,
    "tuple": tuple,
    "len": len,
    "range": range,
    "min": min,
    "max": max,
    "sum": sum,
    "sorted": sorted,
    "enumerate": enumerate,
    "zip": zip,
    "abs": abs,
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "any": any,
    "all": all,
    "round": round,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "filter": filter,
    "map": map,
    "reversed": reversed,
    "print": lambda *args, **kwargs: None,
}

BANNED_NAMES = {
    "eval",
    "exec",
    "compile",
    "__import__",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
}


def create_safe_open(state_str: str = "{}"):
    def _safe_open(file, mode="r", *args, **kwargs):
        if any(m in mode for m in ("w", "a", "+", "x")):
            raise PermissionError("Write access is forbidden in sandbox.")
        return io.StringIO(state_str)
    return _safe_open


def create_safe_input(state_str: str = "{}"):
    def _safe_input(prompt=None):
        return state_str
    return _safe_input


def validate_ast(code_str: str) -> Optional[str]:
    """
    Statically analyzes the AST for dangerous constructs (imports, arbitrary file/system access).
    Returns an error message if invalid, or None if safe.
    """
    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        return f"SyntaxError: {e.msg} at line {e.lineno}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_pkg = alias.name.split(".")[0]
                if root_pkg not in SAFE_MODULES:
                    return f"SecurityViolation: Import of '{alias.name}' is forbidden."
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root_pkg = node.module.split(".")[0]
                if root_pkg not in SAFE_MODULES:
                    return f"SecurityViolation: Import from '{node.module}' is forbidden."
        if isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            return f"SecurityViolation: Use of banned function '{node.id}' is forbidden."
        if isinstance(node, ast.Attribute) and (node.attr.startswith("__") or node.attr in BANNED_NAMES):
            return f"SecurityViolation: Access to internal attribute '{node.attr}' is forbidden."

    return None


def execute_dispatch_code(code_str: str, timeout_sec: float = 2.0, state_str: str = "{}") -> Dict[str, Any]:
    """
    Executes generated Python code inside an AST-validated, thread-isolated sandboxed namespace.
    Enforces AST security checks and hard timeout limits (default: 2.0 seconds).
    Provides in-memory virtual state_str for open(), input(), and pre-parsed telemetry state.
    """
    if not code_str or not code_str.strip():
        return {"ok": False, "error": "EmptyCodeBlock", "plan": None}

    # Auto-wrap raw list literals [ { ... } ] into def get_dispatch_plan()
    code_str_clean = code_str.strip()
    if code_str_clean.startswith("[") and code_str_clean.endswith("]"):
        code_str = f"def get_dispatch_plan():\n    return {code_str_clean}\n"

    # Step 1: Fast static AST security check
    ast_error = validate_ast(code_str)
    if ast_error:
        return {"ok": False, "error": ast_error, "plan": None}

    # Step 2: Thread-isolated execution with strict timeout
    result = {"ok": False, "error": "ExecutionTimeout", "plan": None}

    def _sandboxed_runner():
        nonlocal result
        builtins = dict(SAFE_BUILTINS)
        builtins["open"] = create_safe_open(state_str)
        builtins["input"] = create_safe_input(state_str)
        builtins["__import__"] = lambda name, *args, **kwargs: (
            __import__(name, *args, **kwargs) if name in SAFE_MODULES else (_ for _ in ()).throw(ImportError(f"Import {name} not allowed"))
        )
        parsed_state = {}
        try:
            parsed_state = json.loads(state_str) if state_str else {}
        except Exception:
            pass

        namespace = {
            "__builtins__": builtins,
            "json": json,
            "math": math,
            "data": parsed_state,
            "state": parsed_state,
            "scenario": parsed_state,
            "telemetry": parsed_state,
            "emergency_telemetry": parsed_state,
        }
        try:
            exec(code_str, namespace, namespace)
            if "get_dispatch_plan" not in namespace:
                # Fallback: check if script defined dispatch_plan, plan, or schedule as a list
                for candidate in ("dispatch_plan", "plan", "schedule", "result"):
                    if isinstance(namespace.get(candidate), list):
                        result = {"ok": True, "error": None, "plan": namespace[candidate]}
                        return
                result = {"ok": False, "error": "Missing get_dispatch_plan() function", "plan": None}
                return
            dispatch_fn = namespace["get_dispatch_plan"]
            if not callable(dispatch_fn):
                result = {"ok": False, "error": "get_dispatch_plan is not callable", "plan": None}
                return

            try:
                sig = inspect.signature(dispatch_fn)
                if len(sig.parameters) >= 1:
                    plan = dispatch_fn(parsed_state)
                else:
                    plan = dispatch_fn()
            except (TypeError, ValueError):
                try:
                    plan = dispatch_fn()
                except TypeError:
                    plan = dispatch_fn(parsed_state)

            if not isinstance(plan, list):
                result = {"ok": False, "error": f"Plan did not return a list, got {type(plan).__name__}", "plan": None}
                return

            result = {"ok": True, "error": None, "plan": plan}
        except Exception as exc:
            result = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)}", "plan": None}

    t = threading.Thread(target=_sandboxed_runner)
    t.daemon = True
    t.start()
    t.join(timeout=timeout_sec)

    if t.is_alive():
        return {"ok": False, "error": "ExecutionTimeout", "plan": None}

    return result
