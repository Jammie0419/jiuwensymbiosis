# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Best-effort conversion of arbitrary runtime objects to JSON-safe values.

Shared by the tool-level execution trace (``agent/trace.py``) and the
planning-level episode trace (``agent/episode_trace.py``). Keeping one
implementation means both layers agree on how numpy arrays, dataclasses and
non-finite floats are rendered.

The defaults preserve the historical ``TraceRail`` behaviour: depth 8, numpy
arrays above 64 elements summarised, strings left intact. Callers that write
strict JSONL pass ``finite_only=True`` and ``truncate_strings=True`` so a NaN
or a multi-megabyte string cannot corrupt a line.
"""

from __future__ import annotations

import base64
import math
from typing import Any

import numpy as np

__all__ = ["json_safe"]

_MAX_STRING = 2000
_MAX_ITEMS = 1000
_MAX_DEPTH = 8


def json_safe(
    obj: Any,
    *,
    depth: int = 0,
    max_depth: int = _MAX_DEPTH,
    max_string: int = _MAX_STRING,
    max_items: int = _MAX_ITEMS,
    finite_only: bool = False,
    truncate_strings: bool = False,
) -> Any:
    """Recursively coerce ``obj`` to JSON-serialisable primitives.

    numpy arrays → lists; numpy scalars → py scalars; bytes → base64 str;
    everything else unsupported → ``repr()`` string. Depth-, item- and
    string-bounded so a pathological object cannot blow up a trace file.

    ``finite_only`` maps NaN/inf to ``None`` instead of leaving a value that
    ``json.dumps(..., allow_nan=False)`` would reject; ``truncate_strings``
    caps long strings (the tool-level trace deliberately leaves them intact,
    because its ``output_summary`` already truncates at the end).
    """
    if depth > max_depth:
        return repr(obj)[:max_string]
    if obj is None or isinstance(obj, (bool, int, str)):
        if isinstance(obj, str) and truncate_strings and len(obj) > max_string:
            return obj[:max_string] + f"…<+{len(obj) - max_string} chars>"
        return obj
    if isinstance(obj, float):
        if finite_only and not math.isfinite(obj):
            return None
        return obj
    if isinstance(obj, np.ndarray):
        if obj.size > 64:
            return f"<ndarray shape={obj.shape} dtype={obj.dtype}>"
        return json_safe(
            obj.tolist(),
            depth=depth + 1,
            max_depth=max_depth,
            max_string=max_string,
            max_items=max_items,
            finite_only=finite_only,
            truncate_strings=truncate_strings,
        )
    if isinstance(obj, np.generic):
        return json_safe(
            obj.item(),
            depth=depth,
            max_depth=max_depth,
            max_string=max_string,
            max_items=max_items,
            finite_only=finite_only,
            truncate_strings=truncate_strings,
        )
    if isinstance(obj, bytes):
        try:
            return base64.b64encode(obj).decode("ascii")
        except ValueError:
            return "<bytes>"

    def _one(value: Any) -> Any:
        return json_safe(
            value,
            depth=depth + 1,
            max_depth=max_depth,
            max_string=max_string,
            max_items=max_items,
            finite_only=finite_only,
            truncate_strings=truncate_strings,
        )

    if isinstance(obj, dict):
        items = list(obj.items())
        out = {str(key): _one(value) for key, value in items[:max_items]}
        if len(items) > max_items:
            out["__truncated__"] = len(items) - max_items
        return out
    if isinstance(obj, (list, tuple, set, frozenset)):
        values = list(obj)
        out_list = [_one(value) for value in values[:max_items]]
        if len(values) > max_items:
            out_list.append(f"<+{len(values) - max_items} more>")
        return out_list
    # Objects with __dict__ (dataclasses, plain objects) — best effort.
    if hasattr(obj, "__dict__"):
        try:
            return {key: _one(value) for key, value in vars(obj).items() if not key.startswith("_")}
        except (TypeError, AttributeError, RecursionError):
            return repr(obj)[:max_string]
    return repr(obj)[:max_string]
