"""The answers a brief asks for, read out of a solver's own result.

A brief asks for things like the midspan deflection, the maximum moment and
the support shear forces. The results table carried only peak magnitudes, so
the midspan value and the per-support reactions never appeared by name. This
module pulls them out of either solver's result, with where each peak occurs,
and formats them for a reader who is an engineer, not a programmer:
millimetres, kilonewtons, "downward", "upward".

Nothing here computes a new number from physics. Midspan deflection is the
solver's own interpolation evaluated at midspan, and every other value is
read off the result.
"""

from tools.fem.equation import deflection_at

_SCAN = 2001  # points scanned along a standard-path beam to locate its peak deflection


def _support_kind(flags):
    if flags[1] and flags[5]:
        return "fixed"
    if flags[1]:
        return "pin" if flags[0] else "roller"
    return "rotation held"


def _supports(model, reaction_of):
    out = []
    x_of = {n["id"]: float(n["x"]) for n in model["nodes"]}
    for node, flags in sorted((model.get("supports") or {}).items(), key=lambda kv: x_of[kv[0]]):
        if not (flags[1] or flags[5]):
            continue
        vertical, moment = reaction_of(node, flags)
        out.append({"node": node, "x": x_of[node], "kind": _support_kind(flags), "vertical": vertical, "moment": moment})
    return out


def _equation_results(model, result):
    nodes = sorted(model["nodes"], key=lambda n: n["x"])
    mid = (float(nodes[0]["x"]) + float(nodes[-1]["x"])) / 2
    disp = result["displacements"]
    where = result.get("peak_x") or {}
    x_v = where.get("v")

    def reaction_of(node, flags):
        r = result["reactions"].get(node, {})
        return r.get("F"), (r.get("M") if flags[5] else None)

    return {
        "midspan": {"x": mid, "deflection": deflection_at(model, disp, mid)},
        "max_deflection": {
            "value": result["max_abs"]["v"],
            "x": x_v,
            "deflection": deflection_at(model, disp, x_v) if x_v is not None else None,
        },
        "max_moment": {"value": result["max_abs"]["moment"], "x": where.get("moment")},
        "supports": _supports(model, reaction_of),
    }


def _standard_results(model, result):
    # bending in whichever plane carries the load: y (uy, rz, Mz, FY) or z
    # (uz, ry, My, FZ), with theta_y = -d(uz)/dx in the solver's convention
    in_z = result["max_abs"].get("uz", 0.0) > result["max_abs"].get("uy", 0.0)
    v_key, rot_key, m_key, f_key, sign = ("uz", "ry", "My", "FZ", -1.0) if in_z else ("uy", "rz", "Mz", "FY", 1.0)
    disp = {
        node: {"v": float(d[v_key]), "slope": sign * float(d[rot_key])}
        for node, d in result["displacements"].items()
    }
    nodes = sorted(model["nodes"], key=lambda n: n["x"])
    x0, x1 = float(nodes[0]["x"]), float(nodes[-1]["x"])
    mid = (x0 + x1) / 2
    grid = [x0 + (x1 - x0) * k / (_SCAN - 1) for k in range(_SCAN)]
    values = [abs(deflection_at(model, disp, x)) for x in grid]
    top = max(values)
    x_v = next(x for x, v in zip(grid, values) if v >= top * (1 - 1e-9))  # first from the left
    diagrams = result.get("diagrams") or []
    x_m = None
    if diagrams:
        m_top = max(abs(p[m_key]) for p in diagrams)
        x_m = next(p["x"] for p in diagrams if abs(p[m_key]) >= m_top * (1 - 1e-9))

    def reaction_of(node, flags):
        r = result["reactions"].get(node, {})
        moment_key = "MY" if in_z else "MZ"
        return r.get(f_key), (r.get(moment_key) if flags[5] else None)

    return {
        "midspan": {"x": mid, "deflection": deflection_at(model, disp, mid)},
        "max_deflection": {
            "value": result["max_abs"][v_key],
            "x": x_v,
            "deflection": deflection_at(model, disp, x_v),
        },
        "max_moment": {"value": result["max_abs"][m_key], "x": float(x_m) if x_m is not None else None},
        "supports": _supports(model, reaction_of),
    }


def key_results(model: dict, result: dict, equation: dict = None) -> dict:
    """Midspan deflection, peak deflection and moment with locations, and each
    support's reactions, from either solver's result. SI units, signed: a
    negative deflection is downward, a positive vertical reaction upward."""
    if equation is not None:
        return _equation_results(model, result)
    return _standard_results(model, result)


# ------------------------------------------------------------------ markdown


def _num(value, digits=4):
    return f"{value:.{digits}g}"


def _x(x):
    return "" if x is None else f"x = {_num(x, 3)} m"


def _deflection(v):
    if v is None:
        return "n/a"
    if abs(v) < 1e-12:
        return "0 mm"
    return f"{_num(abs(v) * 1000)} mm {'downward' if v < 0 else 'upward'}"


def key_results_markdown(k: dict) -> str:
    """The Key results section of the report, as markdown."""
    lines = [
        "## Key results",
        "",
        "The answers to the brief, read from the solver's output. Positions are "
        "measured from the left end of the beam.",
        "",
        "| Quantity | Value | Where |",
        "|---|---|---|",
        f"| Midspan deflection | {_deflection(k['midspan']['deflection'])} | {_x(k['midspan']['x'])} |",
        f"| Maximum deflection | {_deflection(k['max_deflection'].get('deflection'))} | {_x(k['max_deflection']['x'])} |",
        f"| Maximum bending moment | {_num(k['max_moment']['value'] / 1000)} kN·m | {_x(k['max_moment']['x'])} |",
    ]
    for s in k["supports"]:
        label = f"{s['node']} ({s['kind']}, x = {_num(s['x'], 3)} m)"
        if s["vertical"] is not None:
            f = s["vertical"]
            direction = "upward" if f > 0 else "downward" if f < 0 else ""
            lines.append(f"| Reaction at {label} | {_num(abs(f) / 1000)} kN {direction}".rstrip() + " | |")
        if s["moment"] is not None:
            lines.append(f"| Moment at {label} | {_num(abs(s['moment']) / 1000)} kN·m | |")
    lines.append("")
    return "\n".join(lines)
