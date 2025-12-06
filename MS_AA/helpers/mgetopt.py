# mgetopt.py
from typing import Any

def mgetopt(opts: Any, varname: str, default: Any, *args: Any) -> Any:
    """
    Python equivalent of MATLAB mgetopt.m
    
    Parameters
    ----------
    opts : dict | object | None
        Container of options. Can be a dict (opts[varname]) or an object with attributes (getattr).
        If None, returns `default`.
    varname : str
        Name of the option to retrieve.
    default : Any
        Value to return if the option is not present.
    *args : Any
        Ignored (kept for signature compatibility with MATLAB varargin).
    
    Returns
    -------
    Any
        opts[varname] / getattr(opts, varname) if present, else `default`.
    """
    if opts is None:
        return default

    # dict-like
    if isinstance(opts, dict):
        return opts.get(varname, default)

    # object-like (e.g., SimpleNamespace or custom config objects)
    if hasattr(opts, varname):
        return getattr(opts, varname)

    # fallback: try mapping protocol without isinstance(dict)
    try:
        return opts[varname]
    except Exception:
        return default
