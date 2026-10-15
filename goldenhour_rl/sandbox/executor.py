import ast
import multiprocessing
from typing import Dict, Any, Optional

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
}

BANNED_AST_NODES = (
    ast.Import,
    ast.ImportFrom,
)

BANNED_NAMES = {
    "open",
    "eval",
    "exec",
    "compile",
    "__import__",
    "input",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
}


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
        if isinstance(node, BANNED_AST_NODES):
            return "SecurityViolation: Imports are strictly forbidden in dispatch plans."
        if isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            return f"SecurityViolation: Use of banned function '{node.id}' is forbidden."
        if isinstance(node, ast.Attribute) and (node.attr.startswith("__") or node.attr in BANNED_NAMES):
            return f"SecurityViolation: Access to internal attribute '{node.attr}' is forbidden."

    return None


def _worker_exec(code_str: str, result_queue: multiprocessing.Queue):
    """Worker process target that executes untrusted code in a sandboxed namespace."""
    namespace = {}
    try:
        exec(code_str, {"__builtins__": SAFE_BUILTINS}, namespace)
        if "get_dispatch_plan" not in namespace:
            result_queue.put({"ok": False, "error": "Missing get_dispatch_plan() function", "plan": None})
            return
        dispatch_fn = namespace["get_dispatch_plan"]
        if not callable(dispatch_fn):
            result_queue.put({"ok": False, "error": "get_dispatch_plan is not callable", "plan": None})
            return
        plan = dispatch_fn()
        result_queue.put({"ok": True, "error": None, "plan": plan})
    except Exception as exc:
        result_queue.put({"ok": False, "error": f"{type(exc).__name__}: {str(exc)}", "plan": None})


def execute_dispatch_code(code_str: str, timeout_sec: float = 2.0) -> Dict[str, Any]:
    """
    Executes generated Python code inside a sandboxed multiprocessing boundary.
    Enforces AST security checks and hard timeout limits (default: 2.0 seconds).
    """
    if not code_str or not code_str.strip():
        return {"ok": False, "error": "EmptyCodeBlock", "plan": None}

    # Step 1: Fast static AST security check
    ast_error = validate_ast(code_str)
    if ast_error:
        return {"ok": False, "error": ast_error, "plan": None}

    # Step 2: Isolated process execution with strict timeout
    queue = multiprocessing.Queue()
    proc = multiprocessing.Process(target=_worker_exec, args=(code_str, queue))
    proc.start()
    proc.join(timeout=timeout_sec)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=0.5)
        return {"ok": False, "error": "ExecutionTimeout", "plan": None}

    if not queue.empty():
        return queue.get()

    return {"ok": False, "error": "SandboxCrash", "plan": None}
