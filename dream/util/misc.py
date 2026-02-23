from typing import Optional, List, Any
import importlib

def deep_merge(orig, new):
    """
    Recursively merge `new` into `orig`. Both `orig` and `new` must be dicts.
    """
    for key, val in new.items():
        existing = orig.get(key)
        if isinstance(existing, dict) and isinstance(val, dict):
            deep_merge(existing, val)
        else:
            orig[key] = val

def head_match(a: str, b: List[str]) -> bool:
    """
    Return True if `a` is a prefix (“head”) of any element in `b`.
    Uses str.startswith (C‐level) and short‐circuits on first match.
    """
    # localize for speed
    startswith = str.startswith
    for s in b:
        if startswith(s, a):
            return True
    return False

def lists_intersection(a: List[Any], b: List[Any]) -> List[Any]:
    """
    Return a list of the unique elements that appear in both a and b,
    in the order they first appear in a.
    """
    b_set = set(b)
    seen = set()
    c: List[Any] = []
    for x in a:
        if x in b_set and x not in seen:
            seen.add(x)
            c.append(x)
    return c


# helper for dynamic or identity function
def mk_func(func_name: Optional[str]):
    if func_name is None:
        return lambda arr, *args, **kwargs: arr
    module_name, fn = func_name.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, fn)


def parse_boolean(value):
    value = value.lower()

    if value in ["True","true", "yes", "y", "1", "t"]:
        return True
    elif value in ["False","false", "no", "n", "0", "f"]:
        return False

    return False


def read_args():
    import argparse
    
    parser = argparse.ArgumentParser(description='preproc')
    parser.add_argument(
        '--exp',
        metavar='NAME',
        type=str,
        default=None,
        help='(optional) name of the experiment'
    )
    parser.add_argument(
        '--run',
        metavar='N',
        type=int,
        default=None,
        help='(optional) run number; if provided, we switch to offline mode'
    )

    args = parser.parse_args()

    exp     = args.exp
    run_num = args.run

    #print(f"exp: {exp!r}, runnum: {run_num!r}")

    mode = 'online' if run_num is None else 'offline'

    return mode, exp, run_num


def nsify(d):
    from types import SimpleNamespace
    if isinstance(d, dict):
        return SimpleNamespace(**{k: nsify(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [nsify(x) for x in d]
    else:
        return d

def read_config(fn,namespace=False):
    import yaml
    with open(fn, "r") as f:
        params = yaml.safe_load(f)
    if namespace: params = nsify(params)    
    return params


def dict_to_yaml_file(data: dict, filepath: str, *, sort_keys: bool = False) -> None:
    import yaml
    with open(filepath, 'w') as f:
        yaml.safe_dump(data, f, sort_keys=sort_keys, default_flow_style=False)