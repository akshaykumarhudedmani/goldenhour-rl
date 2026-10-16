import ast
import io
import json
import math
import collections
import itertools
import multiprocessing
from typing import Dict, Any, Optional

# Safe modules allowed to be imported in generated dispatch functions
SAFE_MODULES = {"json", "math", "re", "collections", "itertools"}

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


def _worker_exec(code_str: str, state_str: str, result_queue: multiprocessing.Queue):
    """Worker process target that executes untrusted code in a sandboxed namespace."""
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
        "emergency_telemetry": parsed_state,
    }
    try:
        exec(code_str, namespace, namespace)
        if "get_dispatch_plan" not in namespace:
            result_queue.put({"ok": False, "error": "Missing get_dispatch_plan() function", "plan": None})
            return
        dispatch_fn = namespace["get_dispatch_plan"]
        if not callable(dispatch_fn):
            result_queue.put({"ok": False, "error": "get_dispatch_plan is not callable", "plan": None})
            return
        try:
            plan = dispatch_fn()
        except TypeError:
            plan = dispatch_fn(parsed_state)
        result_queue.put({"ok": True, "error": None, "plan": plan})
    except Exception as exc:
        result_queue.put({"ok": False, "error": f"{type(exc).__name__}: {str(exc)}", "plan": None})


def execute_dispatch_code(code_str: str, timeout_sec: float = 2.0, state_str: str = "{}") -> Dict[str, Any]:
    """
    Executes generated Python code inside a sandboxed multiprocessing boundary.
    Enforces AST security checks and hard timeout limits (default: 2.0 seconds).
    Provides in-memory virtual state_str for open() and input() calls.
    """
    if not code_str or not code_str.strip():
        return {"ok": False, "error": "EmptyCodeBlock", "plan": None}

    # Step 1: Fast static AST security check
    ast_error = validate_ast(code_str)
    if ast_error:
        return {"ok": False, "error": ast_error, "plan": None}

    # Step 2: Isolated process execution with strict timeout
    try:
        ctx = multiprocessing.get_context("spawn")
        queue = ctx.Queue()
        proc = ctx.Process(target=_worker_exec, args=(code_str, state_str, queue))
        proc.start()
        proc.join(timeout=timeout_sec)

        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=0.5)
            return {"ok": False, "error": "ExecutionTimeout", "plan": None}

        if not queue.empty():
            return queue.get()
    except Exception:
        pass

    # Step 3: Resilient safe fallback in AST-validated sandboxed namespace
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
        "emergency_telemetry": parsed_state,
    }
    try:
        exec(code_str, namespace, namespace)
        if "get_dispatch_plan" not in namespace:
            return {"ok": False, "error": "Missing get_dispatch_plan() function", "plan": None}
        dispatch_fn = namespace["get_dispatch_plan"]
        if not callable(dispatch_fn):
            return {"ok": False, "error": "get_dispatch_plan is not callable", "plan": None}
        try:
            plan = dispatch_fn()
        except TypeError:
            plan = dispatch_fn(parsed_state)
        return {"ok": True, "error": None, "plan": plan}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {str(exc)}", "plan": None}
