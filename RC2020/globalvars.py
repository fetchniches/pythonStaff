from typing import Any


def init_vars():
    global vars_dict 
    vars_dict = {}

def get_var(key: str):
    try:
        return vars_dict[key]
    except KeyError:
        return None

def set_var(key: str, val: Any):
    try:
        return vars_dict[key]
    except KeyError:
        vars_dict.setdefault(key, val)
        return val