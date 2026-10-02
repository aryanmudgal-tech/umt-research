"""Turn a run's tracer events into the professor's six plain-English stages.

The pipeline emits machine events (stage_start, tool_call, gate_check,
verdict, ...). The page shows six stages - reading, modeling, solving,
checking, independent, report - each with a status, a headline, an
explanation a non-programmer can follow, an optional note (a busy model, a
refused equation) and the parameters its animation needs. Narrator.feed()
takes one event and returns the stage updates it causes.

No I/O and no network: the same events replayed give the same updates, which
is what the tests do with a recorded run.
"""

import math

from agent.key_results import key_results

STAGES = ("reading", "modeling", "solving", "checking", "independent", "report")
SOLVERS = {"solve_beam_3d", "solve_with_equation"}

EXPLAIN = {
    "reading": (
        "The AI model reads the brief and extracts the inputs: span, boundary "
        "conditions, loading and the governing equation. It performs no "
        "calculation; every number comes from the finite element solver."
    ),
    "checking": (
        "Deterministic verification before the result is released: global "
        "equilibrium, the residual of the stiffness equations, the boundary "
        "conditions, and a convergence study against a manufactured exact "
        "solution. Any failure means the result cannot be marked verified."
    ),
    "independent": (
        "A second AI model, with no access to the first model's work, re-solves "
        "the problem by a different numerical method and compares the results. "
        "Any disagreement marks the result not verified."
    ),
    "report": (
        "Compiling the report: the key results, the governing equation and its "
        "Galerkin derivation, the full results, and every check with its outcome."
    ),
}

# Every verification check, as a civil engineer would name it: (name, what it tests).
CHECKS = {
    "equilibrium": (
        "Global vertical equilibrium",
        "Support reactions plus any foundation reaction must equal the applied load.",
    ),
    "residual": (
        "Residual of the stiffness equations",
        "K·u − F must vanish at the free degrees of freedom, and the reported reactions "
        "must equal K·u − F at the restrained ones.",
    ),
    "consistent": (
        "Reported results consistent with the solution",
        "Deflections, rotations, moments and shears in the report are recomputed from the "
        "nodal solution and compared.",
    ),
    "symmetric": ("Stiffness matrix symmetry", "K must equal Kᵀ, as it does for this self-adjoint problem."),
    "rigid": (
        "No mechanism",
        "With the boundary conditions applied the structure must be stable: no rigid-body motion.",
    ),
    "supports": ("Boundary conditions satisfied", "Restrained degrees of freedom show zero displacement."),
    "sign": ("Deflection sense", "A downward load must produce a downward deflection."),
    "mms": (
        "Convergence study (manufactured solution)",
        "An exact solution is manufactured for this equation and the load producing it derived; "
        "the finite element solution must converge to it at the theoretical rate, fourth order "
        "for Hermite cubic elements.",
    ),
    "textbook": (
        "Closed-form solution",
        "For a simply supported, uniformly loaded Euler–Bernoulli beam the results must match "
        "δ = 5qL⁴/384EI, M = qL²/8 and V = qL/2.",
    ),
    "pynite": (
        "Independent FEM program (PyNite)",
        "PyNite, an independent frame analysis program, solves the same model and must agree.",
    ),
}
_CHECK_KEY = {
    "equilibrium_forces": "equilibrium",
    "equation_equilibrium": "equilibrium",
    "equation_solution_residual": "residual",
    "equation_samples_consistent": "consistent",
    "stiffness_symmetric": "symmetric",
    "equation_stiffness_symmetry": "symmetric",
    "rigid_body_nullspace": "rigid",
    "supports_respected": "supports",
    "equation_support_conditions": "supports",
    "deflection_negative_under_downward_load": "sign",
    "equation_deflection_sign": "sign",
    "equation_mms": "mms",
    "pynite_agreement": "pynite",
}


def check_key(name: str):
    """The CHECKS key for a gate check's name, or None for one not listed."""
    name = name.removeprefix("invariant_")
    if name.startswith("closed_form"):
        return "textbook"
    return _CHECK_KEY.get(name)


def plain_check_name(name: str) -> str:
    """A gate check's name as a civil engineer would say it."""
    key = check_key(name)
    if key:
        return CHECKS[key][0]
    words = name.removeprefix("invariant_").replace("_", " ").strip()
    return words[:1].upper() + words[1:]


# ------------------------------------------------------------- formatting


def _num(value, digits=4):
    return f"{value:.{digits}g}"


def _force_per_length(w):
    return f"{_num(abs(w) / 1000)} kN/m {'downward' if w < 0 else 'upward'}"


_SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def sci(value: float) -> str:
    """1.5e8 -> '1.5 × 10⁸'; ordinary sizes stay ordinary."""
    if value == 0 or 1e-3 <= abs(value) < 1e5:
        return f"{value:.4g}"
    exponent = math.floor(math.log10(abs(value)))
    return f"{value / 10**exponent:.3g} × 10{str(exponent).translate(_SUPERSCRIPT)}"


def _support_kind(flags):
    if flags[1] and flags[5]:
        return "fixed"
    if flags[1]:
        return "pin" if flags[0] else "roller"
    return "rotation held"


def _numeric(text, params):
    """The value of a coefficient that is a bare number or a single parameter."""
    text = str(text).strip()
    if text in params:
        return float(params[text])
    try:
        return float(text)
    except ValueError:
        return None


# ------------------------------------------------------------ the narrator


class Narrator:
    """Stateful translator from one run's events to stage updates."""

    def __init__(self):
        self.status = {s: "waiting" for s in STAGES}
        self.scene = {"soil": False, "axial": False, "tapered": False, "elements": 0, "span": 0.0, "sag_mm": 0.0}
        self.facts = []
        self.model = None
        self.equation = None
        self.solved = False
        self.checks_seen = 0
        self.failed_checks = []
        self.finished = False
        self.passed = None
        self._last = {}

    # one update for one stage, remembered so a later note can repeat it
    def _update(self, stage, status=None, headline=None, explanation=None, note=None):
        prev = self._last.get(stage, {})
        if status:
            self.status[stage] = status
        update = {
            "stage": stage,
            "status": self.status[stage],
            "headline": headline if headline is not None else prev.get("headline", ""),
            "explanation": explanation if explanation is not None else prev.get("explanation", EXPLAIN.get(stage, "")),
            "note": note,
            "facts": list(self.facts),
            "scene": dict(self.scene),
        }
        self._last[stage] = update
        return update

    def _active(self):
        return next((s for s in STAGES if self.status[s] == "active"), None)

    def feed(self, event) -> list:
        etype, data, stage = event.get("type"), event.get("data") or {}, event.get("stage")
        handler = getattr(self, f"_on_{etype}", None)
        return handler(stage, data) if handler else []

    # ---------------------------------------------------------- handlers

    def _on_stage_start(self, stage, data):
        name = data.get("stage", stage)
        if name == "orchestrator":
            return [self._update("reading", "active", "Reading your brief")]
        if name == "gate":
            return [self._update("checking", "active", "Checking the numbers", note=None)]
        if name == "verifier":
            return [self._update("independent", "active", "Independent check")]
        if name == "report":
            out = []
            if not self.solved:
                out.append(
                    self._update(
                        "reading",
                        "failed",
                        "Couldn't analyze this brief",
                        "The AI model read the brief but did not produce a finite "
                        "element model it could solve. Its reason is shown below.",
                    )
                )
                return out
            out.append(self._update("report", "active", "Writing the report"))
            return out
        return []

    def _on_model_fallback(self, stage, data):
        target = "independent" if data.get("role") == "verifier" else (self._active() or "reading")
        return [self._update(target, note="The AI service is busy; switching to a backup model.")]

    def _on_tool_call(self, stage, data):
        name = data.get("name")
        if stage == "verifier":
            if self.status["independent"] == "active":
                return [self._update("independent", note="Re-solving by an independent numerical method…")]
            return []
        if name not in SOLVERS:
            return []
        args = data.get("args") or {}
        self.model = args.get("model") or {}
        self.equation = args.get("equation") if name == "solve_with_equation" else None
        self._describe()
        n = self.scene["elements"]
        supports = " and ".join(sorted({_support_kind(f) for f in (self.model.get("supports") or {}).values()}))
        dofs = (2 if self.equation is not None else 6) * (n + 1)
        length = f" of {_num(self.scene['span'] / n, 3)} m" if n else ""
        return [
            self._update("reading", "done", self._title()),
            self._update(
                "modeling",
                "done",
                f"{n} elements, {supports or 'no supports'}",
                f"Discretized into {n} Hermite cubic beam elements{length}: {n + 1} "
                f"nodes, {dofs} degrees of freedom.",
            ),
            self._update(
                "solving",
                "active",
                "Solving",
                f"Assembling and solving the global stiffness system K·u = F for "
                f"{self._equation_phrase()}. This is computed by the solver; the AI "
                "model plays no part in it.",
            ),
        ]

    def _on_tool_result(self, stage, data):
        result = data.get("result") or {}
        if stage == "verifier":
            if isinstance(result, dict) and "error" in result:
                return [self._update("independent", note="One of its tools failed, so that comparison verified nothing.")]
            return []
        if data.get("name") not in SOLVERS:
            return []
        if isinstance(result, dict) and "error" in result:
            reason = str(result["error"]).split(":", 1)[-1].strip()
            return [self._update("solving", note=f"The solver rejected this equation ({reason}). The AI model is deciding how to proceed.")]
        self.solved = True
        try:
            k = key_results(self.model, result, self.equation)
        except Exception:
            return [self._update("solving", "done", "Solved")]
        sag = k["midspan"]["deflection"]
        self.scene["sag_mm"] = abs(sag) * 1000
        where = k["max_moment"]["x"]
        explanation = (
            f"Midspan deflection {_num(abs(sag) * 1000)} mm. Maximum bending moment "
            f"{_num(k['max_moment']['value'] / 1000)} kN·m"
            + (f" at x = {_num(where, 3)} m." if where is not None else ".")
        )
        if self.scene["soil"]:
            explanation += " The foundation reaction carries most of the load."
        return [self._update("solving", "done", f"Midspan deflection {_num(abs(sag) * 1000, 3)} mm", explanation)]

    def _on_gate_check(self, stage, data):
        self.checks_seen += 1
        if data.get("status") == "fail":
            self.failed_checks.append(plain_check_name(data.get("name", "")))
        done = f"{self.checks_seen} {'check' if self.checks_seen == 1 else 'checks'} done"
        return [self._update("checking", note=done)]

    def _on_gate_result(self, stage, data):
        applicable = data.get("n_total", 0) - data.get("n_skipped", 0)
        skipped = data.get("n_skipped", 0)
        skipped_text = (
            f" {skipped} {'check does' if skipped == 1 else 'checks do'} not apply to this "
            f"equation and {'was' if skipped == 1 else 'were'} skipped; a skipped check is never "
            "counted as a pass."
            if skipped
            else ""
        )
        if data.get("passed"):
            return [
                self._update(
                    "checking",
                    "done",
                    f"{data.get('n_passed', 0)} of {applicable} checks passed",
                    "All applicable checks passed." + skipped_text,
                )
            ]
        failed = ", ".join(self.failed_checks) or "see the report"
        return [
            self._update(
                "checking",
                "failed",
                f"{data.get('n_failed', 0)} of {applicable} checks failed",
                f"Failed: {failed}. The result cannot be marked verified." + skipped_text,
            )
        ]

    def _on_verdict(self, stage, data):
        if data.get("refuted"):
            reason = (data.get("reasoning") or "").strip() or "it found a problem with the numbers"
            return [self._update("independent", "failed", "The independent check disagreed", f"Its reason: {reason}")]
        return [
            self._update(
                "independent",
                "done",
                "The independent check agrees",
                "The second model re-solved the problem by a different numerical method and its results match.",
            )
        ]

    def _on_run_end(self, stage, data):
        self.finished = True
        self.passed = bool(data.get("passed"))
        if not self.solved:
            return []
        return [self._update("report", "done", "The report is ready", "The report opens below.")]

    # ------------------------------------------------------- descriptions

    def _describe(self):
        model, eq = self.model, self.equation
        nodes = sorted(model.get("nodes") or [], key=lambda n: float(n["x"]))
        span = float(nodes[-1]["x"]) - float(nodes[0]["x"]) if len(nodes) > 1 else 0.0
        self.scene.update(elements=max(len(nodes) - 1, 0), span=span)
        facts = [f"Span: {_num(span, 3)} m"]
        supports = model.get("supports") or {}
        ordered = [_support_kind(supports[n["id"]]) for n in nodes if n["id"] in supports]
        if ordered:
            facts.append("Supports: " + " and ".join(ordered))
        if eq is None:
            for load in model.get("distributed_loads") or []:
                if load.get("w1") == load.get("w2"):
                    facts.append(f"Load: {_force_per_length(load['w1'])}")
                else:
                    facts.append("Load: varying along the span")
        else:
            params = eq.get("params") or {}
            coeffs = eq.get("coeffs") or {}
            q = _numeric(eq.get("rhs", "0"), params)
            facts.append(f"Load: {_force_per_length(q)}" if q else f"Load: {eq.get('rhs')}")
            k = _numeric(coeffs.get("v0", "0"), params)
            if k:
                self.scene["soil"] = True
                facts.append(f"Subgrade modulus: {sci(k)} N/m²")
            P = _numeric(coeffs.get("v2", "0"), params)
            if P:
                self.scene["axial"] = True
                facts.append(f"Axial force: {_num(abs(P) / 1e6)} MN {'compression' if P > 0 else 'tension'}")
            if "x" in str(coeffs.get("v4", "")):
                self.scene["tapered"] = True
                facts.append("EI varies along the span")
        for point in model.get("point_loads") or []:
            if point.get("dof") == "FY" and point.get("value"):
                facts.append(f"Point load: {_num(abs(point['value']) / 1000)} kN at {point['node']}")
        self.facts = facts

    def _title(self):
        label = (self.equation or {}).get("label") or ("Standard beam" if self.equation is None else "Custom equation")
        return f"{label}, {_num(self.scene['span'], 3)} m span"

    def _equation_phrase(self):
        if self.equation is None:
            return "the Euler–Bernoulli beam equation"
        return f"the governing equation from the brief ({(self.equation.get('label') or 'custom').lower()})"
