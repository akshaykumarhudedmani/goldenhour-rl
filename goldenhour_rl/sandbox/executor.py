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
    "TimeoutError": TimeoutError,
    "Exception": Exception,
    "__build_class__": __build_class__,
    "object": object,
    "type": type,
    "super": super,
    "property": property,
    "staticmethod": staticmethod,
    "classmethod": classmethod,
    "True": True,
    "False": False,
    "None": None,
    "true": True,
    "false": False,
    "null": None,
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

BANNED_ATTRS = {
    "__class__",
    "__subclasses__",
    "__globals__",
    "__code__",
    "__bases__",
    "__mro__",
    "__dict__",
    "__closure__",
    "__func__",
    "__self__",
    "__module__",
}

ALLOWED_DUNDERS = {
    "__init__",
    "__str__",
    "__repr__",
    "__len__",
    "__getitem__",
    "__setitem__",
    "__iter__",
    "__next__",
    "__eq__",
    "__ne__",
    "__lt__",
    "__le__",
    "__gt__",
    "__ge__",
    "__contains__",
    "__enter__",
    "__exit__",
    "__bool__",
    "__call__",
    "__hash__",
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
        if isinstance(node, ast.Attribute):
            if node.attr in BANNED_NAMES or node.attr in BANNED_ATTRS:
                return f"SecurityViolation: Access to internal attribute '{node.attr}' is forbidden."
            if node.attr.startswith("__") and node.attr.endswith("__") and node.attr not in ALLOWED_DUNDERS:
                return f"SecurityViolation: Access to internal attribute '{node.attr}' is forbidden."

class LoopBoundingTransformer(ast.NodeTransformer):
    """Injects an iteration counter into all While and For loops to terminate infinite loops in <1ms."""
    def __init__(self, max_iterations: int = 5000):
        self.max_iterations = max_iterations
        self.loop_id = 0

    def visit_While(self, node: ast.While):
        self.generic_visit(node)
        self.loop_id += 1
        counter_name = f"__loop_cnt_{self.loop_id}__"
        init_counter = ast.Assign(
            targets=[ast.Name(id=counter_name, ctx=ast.Store())],
            value=ast.Constant(value=0)
        )
        check_counter = ast.If(
            test=ast.Compare(
                left=ast.Name(id=counter_name, ctx=ast.Load()),
                ops=[ast.Gt()],
                comparators=[ast.Constant(value=self.max_iterations)]
            ),
            body=[
                ast.Raise(
                    exc=ast.Call(
                        func=ast.Name(id="TimeoutError", ctx=ast.Load()),
                        args=[ast.Constant(value="ExecutionTimeout: Loop iteration limit exceeded")],
                        keywords=[]
                    )
                )
            ],
            orelse=[]
        )
        increment_counter = ast.AugAssign(
            target=ast.Name(id=counter_name, ctx=ast.Store()),
            op=ast.Add(),
            value=ast.Constant(value=1)
        )
        node.body.insert(0, check_counter)
        node.body.insert(0, increment_counter)
        return [init_counter, node]

    def visit_For(self, node: ast.For):
        self.generic_visit(node)
        self.loop_id += 1
        counter_name = f"__for_cnt_{self.loop_id}__"
        init_counter = ast.Assign(
            targets=[ast.Name(id=counter_name, ctx=ast.Store())],
            value=ast.Constant(value=0)
        )
        check_counter = ast.If(
            test=ast.Compare(
                left=ast.Name(id=counter_name, ctx=ast.Load()),
                ops=[ast.Gt()],
                comparators=[ast.Constant(value=self.max_iterations)]
            ),
            body=[
                ast.Raise(
                    exc=ast.Call(
                        func=ast.Name(id="TimeoutError", ctx=ast.Load()),
                        args=[ast.Constant(value="ExecutionTimeout: Loop iteration limit exceeded")],
                        keywords=[]
                    )
                )
            ],
            orelse=[]
        )
        increment_counter = ast.AugAssign(
            target=ast.Name(id=counter_name, ctx=ast.Store()),
            op=ast.Add(),
            value=ast.Constant(value=1)
        )
        node.body.insert(0, check_counter)
        node.body.insert(0, increment_counter)
        return [init_counter, node]


class SafeJsonModule:
    """Safe wrapper around json to make loads() idempotent when passed pre-parsed dicts."""
    def __init__(self, real_json):
        self._real_json = real_json

    def loads(self, s, *args, **kwargs):
        if isinstance(s, (dict, list)):
            return s
        if isinstance(s, (str, bytes, bytearray)):
            return self._real_json.loads(s, *args, **kwargs)
        return s

    def __getattr__(self, name):
        return getattr(self._real_json, name)


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

    # Step 2: AST Loop-Bounding Transformation to prevent runaway loops from holding GIL
    try:
        tree = ast.parse(code_str)
        transformer = LoopBoundingTransformer(max_iterations=5000)
        bounded_tree = transformer.visit(tree)
        ast.fix_missing_locations(bounded_tree)
        executable_target = compile(bounded_tree, "<dispatch>", "exec")
    except Exception:
        executable_target = code_str

    # Step 3: Thread-isolated execution with strict timeout
    result = {"ok": False, "error": "ExecutionTimeout", "plan": None}

    def _sandboxed_runner():
        nonlocal result
        builtins = dict(SAFE_BUILTINS)
        safe_json = SafeJsonModule(json)
        builtins["open"] = create_safe_open(state_str)
        builtins["input"] = create_safe_input(state_str)
        builtins["__import__"] = lambda name, *args, **kwargs: (
            safe_json if name == "json" else (
                __import__(name, *args, **kwargs) if name in SAFE_MODULES else (_ for _ in ()).throw(ImportError(f"Import {name} not allowed"))
            )
        )
        parsed_state = {}
        try:
            parsed_state = json.loads(state_str) if state_str else {}
        except Exception:
            pass

        namespace = {
            "__name__": "__dispatch__",
            "__builtins__": builtins,
            "json": safe_json,
            "math": math,
            "True": True,
            "False": False,
            "None": None,
            "true": True,
            "false": False,
            "null": None,
            "data": parsed_state,
            "state": parsed_state,
            "scenario": parsed_state,
            "telemetry": parsed_state,
            "emergency_telemetry": parsed_state,
            "telemetry_data": parsed_state,
            "crisis_state": parsed_state,
            "emergency_state": parsed_state,
        }
        try:
            exec(executable_target, namespace, namespace)
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
        except TimeoutError:
            result = {"ok": False, "error": "ExecutionTimeout", "plan": None}
        except Exception as exc:
            result = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)}", "plan": None}

    t = threading.Thread(target=_sandboxed_runner)
    t.daemon = True
    t.start()
    t.join(timeout=timeout_sec)

    if t.is_alive():
        return {"ok": False, "error": "ExecutionTimeout", "plan": None}

    return result
