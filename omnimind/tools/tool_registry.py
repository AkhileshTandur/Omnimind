import inspect, json
from . import builtin

TOOLS = {
    "calculator": {
        "signature": {"expression":"str"},
        "impl": builtin.calculator,
        "desc": "Evaluate a math expression with math.* support."
    },
}

def list_tools():
    return {k: {"args": v["signature"], "desc": v["desc"]} for k,v in TOOLS.items()}

def call_tool(name, **kwargs):
    spec = TOOLS.get(name)
    if not spec: raise KeyError(f"Unknown tool {name}")
    return spec["impl"](**kwargs)
