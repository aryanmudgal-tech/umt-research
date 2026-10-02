"""How the answer was reached: a run's recorded trace, told to a civil engineer.

build_story(events, meta) turns the tracer events saved with a run into the
numbered steps the professor reads under "How this answer was reached": what
the AI model read, the governing equation and its terms, the finite element
model, the solution, every verification check with what it tests and its
evidence, what the independent check did, and the time taken.

The register is a civil engineer's (elements, degrees of freedom, subgrade
modulus, P-delta, convergence order), never software's (tool calls, JSON,
function names). Everything is read from the recorded events and recomputed
from the recorded solve; nothing is taken from an AI model's prose.
"""

import re

from agent.key_results import key_results
from tools.fem.equation import X, equilibrium_terms, parse_spec
from web.narrator import CHECKS, Narrator, check_key, sci

SOLVERS = {"solve_beam_3d", "solve_with_equation"}
STANDARD_TEX = "EI \\, v'''' = q"

SUPPORT_TEXT = {
    "pin": ("Pin", "Restrains deflection; rotation free."),
    "roller": ("Roller", "Restrains deflection; free to slide and rotate."),
    "fixed": ("Fixed support", "Restrains deflection and rotation."),
    "rotation held": ("Rotational restraint", "Restrains rotation; deflection free."),
}

SKIPPED_WHY = {
    "sign": "Not applicable: with a foundation or an axial term, parts of the beam can legitimately "
    "deflect against the load.",
    "textbook": "Not applicable: no closed-form solution exists for this equation, so the convergence "
    "study above takes its place.",
}

VERIFIER_TOOLS = {
    "independent_equation_check": "re-solved the governing equation by collocation on the strong form, "
    "a different numerical method from the finite element solution",
    "pynite_crosscheck": "re-analysed the model with PyNite, an independent frame analysis program",
    "closed_form_case": "compared against the closed-form solution",
    "run_invariant_checks": "re-ran the equilibrium and stability checks",
}

STOPPED_TITLE = {
    "refused": "Couldn't analyze this brief",
    "busy": "The AI service was busy",
    "error": "Something went wrong",
    "interrupted": "The analysis was interrupted",
}


# --------------------------------------------------------------- numbers


def kn(newtons: float) -> str:
    text = f"{abs(newtons) / 1000:.1f}"
    return text[:-2] if text.endswith(".0") else text


def _m(x: float) -> str:
    return f"{x:.3g} m"


def _seconds(s: float) -> str:
    return "under a second" if s < 1 else f"{s:.0f} s"


def _kn_per_m(w):
    return f"{abs(w) / 1000:.4g} kN/m, {'downward' if w < 0 else 'upward'}"


# ----------------------------------------------------------------- events


class _Run:
    """The facts a story needs, pulled out of the event list once."""

    def __init__(self, events):
        self.events = events
        self.attempts = {"orchestrator": [], "verifier": []}
        self.busy = {"orchestrator": [], "verifier": []}
        self.call = self.result = self.call_t = self.result_t = None
        self.checks, self.gate_result, self.verdict = [], None, None
        self.verifier_tools, self.end = [], None
        for e in events:
            data, t = e.get("data") or {}, e.get("t", 0.0)
            kind = e.get("type")
            if kind == "model_attempt" and data.get("role") in self.attempts:
                self.attempts[data["role"]].append(data.get("model"))
            elif kind == "model_fallback" and data.get("role") in self.busy:
                self.busy[data["role"]].append(data.get("model"))
            elif kind == "tool_call" and e.get("stage") == "orchestrator" and data.get("name") in SOLVERS:
                self.call, self.call_t = data, t
            elif kind == "tool_result" and e.get("stage") == "orchestrator" and data.get("name") in SOLVERS:
                if isinstance(data.get("result"), dict) and "error" not in data["result"]:
                    self.result, self.result_t = data["result"], t
            elif kind == "tool_call" and e.get("stage") == "verifier":
                self.verifier_tools.append(data.get("name"))
            elif kind == "gate_check":
                self.checks.append(data)
            elif kind == "gate_result":
                self.gate_result = data
            elif kind == "verdict":
                self.verdict = data
            elif kind == "run_end":
                self.end = data

    @property
    def model(self):
        return (self.call or {}).get("args", {}).get("model") or {}

    @property
    def equation(self):
        if (self.call or {}).get("name") != "solve_with_equation":
            return None
        return (self.call or {}).get("args", {}).get("equation")


def _step(key, title, summary, status="done", **extra):
    return dict({"key": key, "title": title, "summary": summary, "status": status, "items": []}, **extra)


def _support_kind(flags):
    if flags[1] and flags[5]:
        return "fixed"
    if flags[1]:
        return "pin" if flags[0] else "roller"
    return "rotation held"


def _value_of(expr):
    return float(expr) if X not in expr.free_symbols else None


# ------------------------------------------------------------------ steps


def _reading(run):
    used = (run.attempts["orchestrator"] or ["an AI model"])[-1]
    summary = (
        f"The AI model {used} read the brief and extracted the inputs below. It performs no "
        "calculation: every number in the report comes from the finite element solver."
    )
    busy = run.busy["orchestrator"]
    if busy:
        names = " and ".join(busy)
        summary += f" {names} {'was' if len(busy) == 1 else 'were'} busy, so {used} read the brief instead."
    narrator = Narrator()
    for event in run.events:
        narrator.feed(event)
    step = _step("reading", "Read the brief", summary)
    step["items"] = [{"text": fact} for fact in narrator.facts]
    return step


def _equation(run):
    eq = run.equation
    if eq is None:
        section, material = run.model.get("section", {}), run.model.get("material", {})
        step = _step(
            "equation",
            "Governing equation",
            "Euler–Bernoulli beam equation: the brief describes a prismatic beam with no foundation, "
            "axial load or varying section.",
            tex=STANDARD_TEX,
        )
        if material.get("E") and section.get("Iz"):
            step["items"].append({"text": f"Flexural rigidity EI = {sci(material['E'] * section['Iz'])} N·m²"})
        for load in run.model.get("distributed_loads") or []:
            if load.get("w1") == load.get("w2"):
                step["items"].append({"text": f"Distributed load q = {_kn_per_m(load['w1'])}"})
        return step

    import sympy as sp

    from agent.run_phase1 import _residual_latex  # one equation printer for report and story
    from tools.fem.derivation import validate_equation_spec

    parsed = validate_equation_spec(eq)
    tex = f"{_residual_latex(parsed['coeffs'])} = {sp.latex(parsed['rhs'])}"
    label = eq.get("label") or "a custom equation"
    step = _step("equation", "Governing equation", f"Taken from the brief: {label}.", tex=tex)

    numeric = parse_spec(eq)
    nodes = sorted(run.model.get("nodes") or [], key=lambda n: float(n["x"]))
    x0, x1 = (float(nodes[0]["x"]), float(nodes[-1]["x"])) if nodes else (0.0, 0.0)
    a4, a2, a1, a0 = (numeric.coeffs[k] for k in ("v4", "v2", "v1", "v0"))
    if _value_of(a4) is not None:
        step["items"].append({"text": f"Flexural rigidity EI = {sci(_value_of(a4))} N·m²"})
    else:
        left, right = float(a4.subs(X, x0)), float(a4.subs(X, x1))
        step["items"].append({
            "text": f"Flexural rigidity varies along the span: EI(x) = {eq['coeffs']['v4']}",
            "detail": f"From {sci(left)} N·m² at the left support to {sci(right)} N·m² at the right.",
        })
    if not a2.is_zero:
        P = _value_of(a2)
        text = (
            f"Axial force P = {sci(abs(P))} N, {'compression' if P > 0 else 'tension'} (the P–Δ term)"
            if P is not None
            else f"Axial force varies along the span: {eq['coeffs']['v2']}"
        )
        step["items"].append({"text": text})
    if not a1.is_zero:
        step["items"].append({"text": f"First-derivative term: {eq['coeffs']['v1']}"})
    if not a0.is_zero:
        k = _value_of(a0)
        text = (
            f"Modulus of subgrade reaction k = {sci(k)} N/m² (Winkler foundation)"
            if k is not None
            else f"Subgrade modulus varies along the span: {eq['coeffs']['v0']}"
        )
        step["items"].append({"text": text})
    q = _value_of(numeric.rhs)
    step["items"].append(
        {"text": f"Distributed load q = {_kn_per_m(q)}" if q is not None else f"Distributed load q(x) = {eq.get('rhs')}"}
    )
    return step


def _model(run):
    nodes = sorted(run.model.get("nodes") or [], key=lambda n: float(n["x"]))
    n = max(len(nodes) - 1, 0)
    per_node = 2 if run.equation is not None else 6
    lengths = {round(float(b["x"]) - float(a["x"]), 9) for a, b in zip(nodes, nodes[1:])}
    size = f" of {_m(lengths.pop())}" if len(lengths) == 1 else " of varying length"
    kind = "Hermite cubic beam elements" if run.equation is not None else "3D frame elements"
    step = _step(
        "model",
        "Finite element model",
        f"Discretized into {n} {kind}{size}: {len(nodes)} nodes, {per_node * len(nodes)} degrees of freedom.",
    )
    supports = run.model.get("supports") or {}
    for node in nodes:
        flags = supports.get(node["id"])
        if flags and (flags[1] or flags[5]):
            name, what = SUPPORT_TEXT[_support_kind(flags)]
            step["items"].append({"text": f"{name} at x = {_m(float(node['x']))}", "detail": what})
    for load in run.model.get("point_loads") or []:
        if load.get("dof") == "FY" and load.get("value"):
            step["items"].append({"text": f"Point load of {kn(load['value'])} kN at {load['node']}"})
    return step


def _solving(run):
    nodes = run.model.get("nodes") or []
    per_node = 2 if run.equation is not None else 6
    took = (run.result_t or 0.0) - (run.call_t or 0.0)
    step = _step(
        "solving",
        "Solution",
        f"The global stiffness system K·u = F ({per_node * len(nodes)} degrees of freedom) was assembled "
        f"from the Galerkin weak form and solved directly in {_seconds(took)}; the AI model played no part. "
        "Bending moment and shear are recovered from the element end forces and element equilibrium.",
    )
    try:
        k = key_results(run.model, run.result, run.equation)
    except Exception:
        return step
    from agent.key_results import _deflection

    step["items"].append({"text": f"Midspan deflection: {_deflection(k['midspan']['deflection'])}"})
    peak = k["max_deflection"]
    if peak.get("x") is not None:
        step["items"].append({"text": f"Maximum deflection: {_deflection(peak.get('deflection'))} at x = {_m(peak['x'])}"})
    moment = k["max_moment"]
    where = f" at x = {_m(moment['x'])}" if moment.get("x") is not None else ""
    step["items"].append({"text": f"Maximum bending moment: {moment['value'] / 1000:.4g} kN·m{where}"})
    for s in k["supports"]:
        if s["vertical"] is not None:
            direction = "upward" if s["vertical"] > 0 else "downward"
            step["items"].append(
                {"text": f"Reaction at the {s['kind']} (x = {_m(s['x'])}): {kn(s['vertical'])} kN, {direction}"}
            )
    terms = _balance(run)
    if terms and terms["soil"] > 1e-6 * terms["load"]:
        step["items"].append({
            "text": f"Load sharing: the foundation carries {kn(terms['soil'])} kN of the {kn(terms['load'])} kN "
            f"applied load; the supports carry {kn(terms['supports'])} kN."
        })
    return step


def _balance(run):
    """Supports, foundation and total load in N, as magnitudes, or None."""
    try:
        if run.equation is not None:
            t = equilibrium_terms(run.model, run.equation, run.result)
            return {"supports": t["reactions"], "soil": -t["carried"], "load": -(t["distributed"] + t["point_loads"])}
        supports = sum(r.get("FY", 0.0) for r in run.result["reactions"].values())
        return {"supports": supports, "soil": 0.0, "load": -run.result["total_applied"]["FY"]}
    except Exception:
        return None


def _evidence(run, key, check):
    detail = check.get("detail", "")
    if key == "equilibrium":
        t = _balance(run)
        if t is None:
            return None
        if t["soil"] > 1e-6 * t["load"]:
            return (
                f"Supports {kn(t['supports'])} kN + foundation {kn(t['soil'])} kN = "
                f"{kn(t['supports'] + t['soil'])} kN = applied load."
            )
        return f"Supports {kn(t['supports'])} kN = applied load."
    if key == "mms":
        meshes = re.search(r"meshes \(([\d, ]+)\)", detail)
        order = re.search(r"observed order ([\d.]+)", detail)
        if meshes and order:
            sizes = [m.strip() for m in meshes.group(1).split(",")]
            listed = ", ".join(sizes[:-1]) + f" and {sizes[-1]}" if len(sizes) > 1 else sizes[0]
            p = float(order.group(1))
            return (
                f"Meshes of {listed} elements: the error fell {2 ** p:.0f}× per halving of the element size, "
                f"an observed order of {p:.2f} against the theoretical 4."
            )
    if key == "residual":
        left = re.search(r"is ([0-9.]+e[-+]\d+)", detail)
        if left:
            return f"Relative residual {sci(float(left.group(1)))}: zero to machine precision."
    return None


def _checking(run):
    items = []
    for check in run.checks:
        name = check.get("name", "")
        key = check_key(name)
        title, what = CHECKS.get(key, (name.removeprefix("invariant_").replace("_", " ").capitalize(), ""))
        status = check.get("status", "pass")
        if status == "skipped":
            evidence = SKIPPED_WHY.get(key, "Not applicable to this model.")
        else:
            evidence = _evidence(run, key, check)
        items.append({"text": title, "detail": what, "status": status, "evidence": evidence})
    gate = run.gate_result or {}
    applicable = gate.get("n_total", len(items)) - gate.get("n_skipped", 0)
    passed = gate.get("n_passed", sum(1 for i in items if i["status"] == "pass"))
    step = _step(
        "checking",
        "Verification checks",
        f"Deterministic checks run before the result is released: {passed} of {applicable} applicable "
        "checks passed. A check that does not apply is reported as skipped and is never counted as a pass.",
        status="done" if gate.get("passed", True) else "failed",
    )
    step["items"] = items
    return step


def _independent(run):
    verdict = run.verdict or {}
    model = verdict.get("model") or (run.attempts["verifier"] or ["a second AI model"])[-1]
    did = [VERIFIER_TOOLS.get(name, f"used {name}") for name in dict.fromkeys(run.verifier_tools)]
    if verdict.get("model") is None and not run.verifier_tools and verdict.get("refuted"):
        return _step(
            "independent",
            "Independent check",
            "The second AI model could not run because the AI service was busy, so the result is not verified.",
            status="failed",
            verdict="Not verified.",
        )
    summary = f"A second AI model, {model}, with no access to the first model's work, reviewed the result. "
    summary += (
        f"It {' and '.join(did)}, and compared the results."
        if did
        else "It used none of its analysis tools, so its review rests on its own reading of the results."
    )
    busy = run.busy["verifier"]
    if busy:
        summary += f" {' and '.join(busy)} {'was' if len(busy) == 1 else 'were'} busy, so {model} did the review."
    refuted = bool(verdict.get("refuted"))
    reason = (verdict.get("reasoning") or "").strip()
    step = _step(
        "independent",
        "Independent check",
        summary,
        status="failed" if refuted else "done",
        verdict=(f"Result disputed. {reason}" if refuted else "Result confirmed: no discrepancy found."),
    )
    step["items"] = [
        {"text": c.get("name", "Check"), "status": "pass" if c.get("passed") else "fail", "evidence": c.get("detail")}
        for c in verdict.get("checks") or []
    ]
    return step


def build_story(events: list, meta: dict) -> dict:
    """The steps of a run, in order, plus its outcome."""
    run = _Run(events or [])
    status = (meta or {}).get("status", "error")
    steps = [_reading(run)]
    if run.result is not None:
        steps += [_equation(run), _model(run), _solving(run)]
        if run.checks:
            steps.append(_checking(run))
        if run.verdict is not None:
            steps.append(_independent(run))
    if status in ("passed", "failed"):
        duration = (meta or {}).get("duration_s") or (run.end or {}).get("duration_s") or 0.0
        steps.append(_step("report", "Report", f"Report compiled; total analysis time {duration:.0f} s."))
    else:
        steps.append(_step("stopped", STOPPED_TITLE.get(status, "Stopped"), (meta or {}).get("message", ""), status="failed"))
    return {"steps": steps, "outcome": {"status": status, "message": (meta or {}).get("message", "")}}
