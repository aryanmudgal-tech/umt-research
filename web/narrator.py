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

from agent.key_results import key_results

STAGES = ("reading", "modeling", "solving", "checking", "independent", "report")
SOLVERS = {"solve_beam_3d", "solve_with_equation"}

EXPLAIN = {
    "reading": (
        "The AI is reading your brief and picking out the facts the analysis "
        "needs: the span, how the beam is held up, the load on it, and which "
        "equation governs it. It only reads here; it doesn't calculate anything."
    ),
    "checking": (
        "Plain computer checks test the answer before anyone sees it: do the "
        "forces balance, do the supports stay put, does the answer satisfy the "
        "equations exactly, and does the method get a known answer right? Any "
        "failure means the result can't be marked verified."
    ),
    "independent": (
        "A second AI, which never saw the first one's work, solves the same "
        "problem by a different method and compares the numbers. If they "
        "disagree, the result is marked not verified."
    ),
    "report": (
        "Everything goes into one report: the answers to your brief, the "
        "equation that was solved, how it was derived, the full results, and "
        "every check with its outcome."
    ),
}

_PLAIN_CHECKS = {
    "equilibrium_forces": "Forces balance",
    "equation_equilibrium": "Forces balance",
    "equation_solution_residual": "The answer satisfies the equations",
    "equation_samples_consistent": "Reported numbers match the solution",
    "stiffness_symmetric": "Stiffness matrix is symmetric",
    "equation_stiffness_symmetry": "Stiffness matrix is symmetric",
    "rigid_body_nullspace": "The beam can't move without bending",
    "supports_respected": "Supports stay put",
    "equation_support_conditions": "Supports stay put",
    "deflection_negative_under_downward_load": "The beam bends the way it's pushed",
    "equation_deflection_sign": "The beam bends the way it's pushed",
    "equation_mms": "The method gets a known answer right",
    "pynite_agreement": "A second FEM program agrees",
}


def plain_check_name(name: str) -> str:
    """A gate check's name as a professor would say it."""
    name = name.removeprefix("invariant_")
    if name.startswith("closed_form"):
        return "Matches the textbook formula"
    if name in _PLAIN_CHECKS:
        return _PLAIN_CHECKS[name]
    words = name.replace("_", " ").strip()
    return words[:1].upper() + words[1:]


# ------------------------------------------------------------- formatting


def _num(value, digits=4):
    return f"{value:.{digits}g}"


def _force_per_length(w):
    return f"{_num(abs(w) / 1000)} kN/m {'downward' if w < 0 else 'upward'}"


def _stiffness(k):
    if abs(k) >= 1e6:
        return f"{_num(k / 1e6)} million N/m²"
    return f"{_num(k)} N/m²"


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
                        "The AI read the brief but did not produce a beam model it "
                        "could solve. Its reason is shown below.",
                    )
                )
                return out
            out.append(self._update("report", "active", "Writing the report"))
            return out
        return []

    def _on_model_fallback(self, stage, data):
        target = "independent" if data.get("role") == "verifier" else (self._active() or "reading")
        return [self._update(target, note="The AI service is busy, so it is switching to a backup model.")]

    def _on_tool_call(self, stage, data):
        name = data.get("name")
        if stage == "verifier":
            if self.status["independent"] == "active":
                return [self._update("independent", note="Solving the beam again by a different method…")]
            return []
        if name not in SOLVERS:
            return []
        args = data.get("args") or {}
        self.model = args.get("model") or {}
        self.equation = args.get("equation") if name == "solve_with_equation" else None
        self._describe()
        n = self.scene["elements"]
        supports = " and ".join(sorted({_support_kind(f) for f in (self.model.get("supports") or {}).values()}))
        return [
            self._update("reading", "done", self._title()),
            self._update(
                "modeling",
                "done",
                f"{n} pieces, {supports or 'no supports'}",
                f"The beam is split into {n} short pieces joined end to end. Each "
                "piece is simple to describe, and together they behave like the "
                "whole beam: this is the finite element method.",
            ),
            self._update(
                "solving",
                "active",
                "Solving",
                f"The computer, not the AI, now solves {self._equation_phrase()} for "
                f"all {n} pieces at once, working out how far every point of the "
                "beam moves.",
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
            return [self._update("solving", note=f"The solver couldn't take this equation ({reason}). The AI is deciding what to do.")]
        self.solved = True
        try:
            k = key_results(self.model, result, self.equation)
        except Exception:
            return [self._update("solving", "done", "Solved")]
        sag = k["midspan"]["deflection"]
        self.scene["sag_mm"] = abs(sag) * 1000
        where = k["max_moment"]["x"]
        explanation = (
            f"The beam sinks {_num(abs(sag) * 1000)} mm at midspan. The largest "
            f"bending moment is {_num(k['max_moment']['value'] / 1000)} kN·m"
            + (f", {_num(where, 3)} m from the left end." if where is not None else ".")
        )
        if self.scene["soil"]:
            explanation += " The soil pushes back along the whole beam and carries most of the load."
        return [self._update("solving", "done", f"Midspan sag {_num(abs(sag) * 1000, 3)} mm", explanation)]

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
            f" {skipped} {'check' if skipped == 1 else 'checks'} didn't apply to this "
            "equation and were skipped, which is never counted as a pass."
            if skipped
            else ""
        )
        if data.get("passed"):
            return [
                self._update(
                    "checking",
                    "done",
                    f"{data.get('n_passed', 0)} of {applicable} checks passed",
                    "Every check that applies to this beam passed." + skipped_text,
                )
            ]
        failed = ", ".join(self.failed_checks) or "see the report"
        return [
            self._update(
                "checking",
                "failed",
                f"{data.get('n_failed', 0)} of {applicable} checks failed",
                f"Failed: {failed}. The result can't be marked verified." + skipped_text,
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
                "A second AI solved the same problem by a different method and its numbers match ours.",
            )
        ]

    def _on_run_end(self, stage, data):
        self.finished = True
        self.passed = bool(data.get("passed"))
        if not self.solved:
            return []
        return [self._update("report", "done", "Your report is ready", "The report opens below.")]

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
                facts.append(f"Soil stiffness: {_stiffness(k)}")
            P = _numeric(coeffs.get("v2", "0"), params)
            if P:
                self.scene["axial"] = True
                facts.append(f"Axial force: {_num(abs(P) / 1e6)} MN {'compression' if P > 0 else 'tension'}")
            if "x" in str(coeffs.get("v4", "")):
                self.scene["tapered"] = True
                facts.append("Stiffness varies along the span")
        for point in model.get("point_loads") or []:
            if point.get("dof") == "FY" and point.get("value"):
                facts.append(f"Point load: {_num(abs(point['value']) / 1000)} kN at {point['node']}")
        self.facts = facts

    def _title(self):
        label = (self.equation or {}).get("label") or ("Standard beam" if self.equation is None else "Custom equation")
        return f"{label}, {_num(self.scene['span'], 3)} m span"

    def _equation_phrase(self):
        if self.equation is None:
            return "the standard beam equation"
        return f"your equation ({(self.equation.get('label') or 'custom').lower()})"
