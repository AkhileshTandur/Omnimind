import math, json

def calculator(expression: str):
    # intentionally constrained eval
    allowed = {k:getattr(math,k) for k in dir(math) if not k.startswith("_")}
    allowed["__builtins__"] = {}
    return {"result": eval(expression, allowed, {})}
