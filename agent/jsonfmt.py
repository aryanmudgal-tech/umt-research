"""Display formatter for the JSON the trace carries.

json.dumps(indent=2) gives every array element its own line, so a six-flag
support vector becomes twelve lines and a 24-point diagram array becomes
hundreds - the model dict, the one thing a reader most wants to see, ends up
the least readable block in the run. compact_json keeps primitive arrays and
small objects on one line and folds long object arrays behind a count.

This is a display formatter, not a serializer: the fold marker is prose, so
the output is meant to be read, not fed back to json.loads. Both the HTML
viewer and the terminal view render through it, so the two agree line for line.
"""

import json
import re

FOLD_MARKER = "... {hidden} more items ({total} total)"

_FOLD_RE = re.compile(r"^\s*\.\.\. \d+ more items \(\d+ total\)$", re.MULTILINE)


def compact_json(obj, *, indent=2, inline_width=88, fold_after=4) -> str:
    """Render `obj` as indented JSON-shaped text that stays short.

    Args:
        obj: any JSON-ish value; anything json cannot encode becomes str().
        indent: spaces per nesting level.
        inline_width: the widest line the one-line forms may produce.
        fold_after: how many elements of an array of containers to show
            before the rest are replaced by a single marker line. A value
            that already fits on one line is never folded - folding exists
            to stop a block exploding down the page, and one line is no
            explosion.

    Returns:
        Text, newline-separated and never trailing-newline terminated. Key
        order is preserved. Folding loses elements on purpose, so the result
        does not necessarily parse.
    """
    pad = " " * max(0, int(indent))
    width = max(0, int(inline_width))
    fold = max(0, int(fold_after))
    return "\n".join(_render(obj, 0, "", pad, width, fold))


def was_folded(text: str) -> bool:
    """True when compact_json dropped elements from `text`."""
    return bool(_FOLD_RE.search(text or ""))


def _is_primitive(value) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def _scalar(value) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _key(key) -> str:
    return json.dumps(str(key), ensure_ascii=False)


def _inline(value) -> str:
    """The whole value on one line, however long that turns out to be."""
    if isinstance(value, dict):
        return "{" + ", ".join(f"{_key(k)}: {_inline(v)}" for k, v in value.items()) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_inline(v) for v in value) + "]"
    return _scalar(value)


def _render(value, depth, prefix, pad, width, fold, tail="") -> list:
    """Lines for one value. `tail` is the separator the parent owes it."""
    head = pad * depth + prefix
    if not isinstance(value, (dict, list, tuple)):
        return [head + _scalar(value) + tail]

    flat = _inline(value)
    if not value or len(head) + len(flat) + len(tail) <= width:
        return [head + flat + tail]

    if isinstance(value, dict):
        return _object_lines(value, head, depth, pad, width, fold, tail)
    return _array_lines(list(value), head, depth, pad, width, fold, tail)


def _object_lines(value, head, depth, pad, width, fold, tail) -> list:
    lines = [head + "{"]
    keys = list(value.keys())
    for index, key in enumerate(keys):
        lines.extend(
            _render(
                value[key],
                depth + 1,
                _key(key) + ": ",
                pad,
                width,
                fold,
                "," if index < len(keys) - 1 else "",
            )
        )
    lines.append(pad * depth + "}" + tail)
    return lines


def _array_lines(items, head, depth, pad, width, fold, tail) -> list:
    if all(_is_primitive(item) for item in items):
        return _wrapped(items, head, depth, pad, width, tail)

    # Folding costs a marker line, so it only pays off past a couple of items.
    # The margin also keeps `hidden` at 3 or more, so the marker is never singular.
    shown = items[:fold] if len(items) > fold + 2 else items
    hidden = len(items) - len(shown)
    lines = [head + "["]
    for index, item in enumerate(shown):
        last = index == len(shown) - 1 and not hidden
        lines.extend(_render(item, depth + 1, "", pad, width, fold, "" if last else ","))
    if hidden:
        lines.append(
            pad * (depth + 1) + FOLD_MARKER.format(hidden=hidden, total=len(items))
        )
    lines.append(pad * depth + "]" + tail)
    return lines


def _wrapped(items, head, depth, pad, width, tail) -> list:
    """Primitives packed as few lines as the width allows - never one each."""
    parts = [_scalar(item) for item in items]
    inner = pad * (depth + 1)
    lines = [head + "["]
    current = ""
    for index, part in enumerate(parts):
        piece = part + ("," if index < len(parts) - 1 else "")
        if not current:
            current = inner + piece
        elif len(current) + 1 + len(piece) <= width:
            current += " " + piece
        else:
            lines.append(current)
            current = inner + piece
    lines.append(current)
    lines.append(pad * depth + "]" + tail)
    return lines
