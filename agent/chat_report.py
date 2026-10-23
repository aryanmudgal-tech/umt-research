"""The answer the professor reads in the chat: the run's own report, shortened.

On Agent Platform the professor reads the agent's reply in a chat window, not
the formatted report page, so this cuts the report down to what he asked for:
the verdict, the key results, the equation that was solved, and how each check
went. Every number is copied from the report the pipeline wrote, which reads
them from the solver's output; nothing here computes. The full report, with the
model, the derivation and every check's detail, stays one request away.

Plain Markdown only: no typeset math and no raw data blocks, which a chat
window may show as code.
"""

import re

from agent.check_names import plain_check_name
from agent.gates import check_status

FULL_REPORT_HINT = (
    "Ask for the full report to see the beam model, the derivation of the "
    "element equations and the detail of every check."
)

_DERIVATIVES = (("v4", "v''''"), ("v2", "v''"), ("v1", "v'"), ("v0", "v"))


def section(report: str, title: str) -> str:
    """The '## title' section of a report, heading included, or ''."""
    lines = (report or "").splitlines()
    try:
        start = lines.index(f"## {title}")
    except ValueError:
        return ""
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines)
    )
    return "\n".join(lines[start:end]).strip()


def _top_level_sum(text: str) -> bool:
    """True if text adds or subtracts outside any parentheses (a leading sign aside)."""
    depth = 0
    for i, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char in "+-" and depth == 0 and i > 0 and text[i - 1] not in "eE^(":
            return True
    return False


def _math(expr) -> str:
    """A spec expression as an engineer writes it: E*I -> E·I, L**2 -> L^2."""
    text = str(expr).strip().replace("**", "^").replace("*", "·")
    return f"({text})" if _top_level_sum(text) else text


def _is_zero(expr) -> bool:
    return expr is None or re.fullmatch(r"\s*0*\.?0*\s*", str(expr)) is not None


def equation_line(equation: dict) -> str:
    """a4 v'''' + a2 v'' + a1 v' + a0 v = f in plain text, as the solver reads it.

    A coefficient that varies along the span is written in the divergence form
    the elements integrate, (EI(x)·v'')'', since that is the equation solved.
    """
    coeffs = equation.get("coeffs") or {}
    terms = []
    for key, derivative in _DERIVATIVES:
        coeff = coeffs.get(key)
        if _is_zero(coeff):
            continue
        varies = re.search(r"\bx\b", str(coeff)) is not None
        if varies and key == "v4":
            terms.append(f"({_math(coeff)}·v'')''")
        elif varies and key == "v2":
            terms.append(f"({_math(coeff)}·v')'")
        else:
            terms.append(f"{_math(coeff)}·{derivative}")
    rhs = equation.get("rhs")
    return f"{' + '.join(terms) or '0'} = {'0' if _is_zero(rhs) else _math(rhs)}"


def _equation_lines(equation) -> list:
    if not equation:
        return []
    params = ", ".join(f"{name} = {value:.6g}" for name, value in (equation.get("params") or {}).items())
    lines = [
        "## Governing equation",
        "",
        f"{equation.get('label') or 'Equation from the brief'}:",
        "",
        f"    {equation_line(equation)}",
        "",
    ]
    if params:
        lines += [f"with {params} (SI units; deflection v is downward negative).", ""]
    return lines


def _check_lines(gate) -> list:
    if not gate or not gate.get("checks"):
        return []
    checks = gate["checks"]
    statuses = [check_status(c) for c in checks]
    passed = statuses.count("pass")
    skipped = statuses.count("skipped")
    lines = [
        "## Engineering checks",
        "",
        f"{passed} of {len(checks)} passed"
        + (f", {skipped} not applicable to this beam" if skipped else "")
        + ".",
        "",
        "| Check | Result |",
        "|---|---|",
    ]
    for check, status in zip(checks, statuses):
        result = {"pass": "Passed", "fail": "FAILED", "skipped": "Not applicable"}[status]
        lines.append(f"| {plain_check_name(check['name'])} | {result} |")
    failed = [(c, s) for c, s in zip(checks, statuses) if s == "fail"]
    if failed:
        lines += ["", "Why the failed checks failed:", ""]
        lines += [f"- {plain_check_name(c['name'])}: {c.get('detail', '')}" for c, _ in failed]
    return lines + [""]


def _review_lines(verdict) -> list:
    if not verdict:
        return []
    found = "found a problem" if verdict.get("refuted") else "found no error"
    lines = [
        "## Independent review",
        "",
        f"A second model, working on its own, re-checked the results and {found}.",
        "",
    ]
    checks = verdict.get("checks") or []
    if checks:
        lines += ["| What it checked | Result | Detail |", "|---|---|---|"]
        for check in checks:
            result = "Agrees" if check.get("passed") else "DISAGREES"
            detail = str(check.get("detail", "")).replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {check.get('name', '')} | {result} | {detail} |")
        lines.append("")
    if verdict.get("refuted") and verdict.get("reasoning"):
        lines += [f"Its reasoning: {verdict['reasoning']}", ""]
    return lines


def _headline(outcome) -> str:
    if outcome.passed:
        return (
            "**Verified.** Every engineering check passed and the independent "
            "review found no error."
        )
    reasons = []
    gate = outcome.gate or {}
    failed = [c for c in gate.get("checks", []) if check_status(c) == "fail"]
    if failed:
        reasons.append(f"{len(failed)} engineering check{'s' if len(failed) > 1 else ''} failed")
    if (outcome.verdict or {}).get("refuted"):
        reasons.append("the independent review found a problem")
    why = " and ".join(reasons) or "the checks could not confirm it"
    return f"**Not verified:** {why}. Do not rely on these numbers until the cause is found."


def chat_report(outcome, report: str = "") -> str:
    """The chat answer for one finished run, from its RunOutcome and report text."""
    if outcome.code == 2:  # no solver was called: say why, in the orchestrator's words
        return "\n".join(
            [
                "**No analysis was run.**",
                "",
                outcome.narrative.strip() or "The brief could not be turned into a beam model.",
                "",
                "Change the brief and send it again.",
                "",
            ]
        )
    lines = [_headline(outcome), ""]
    key = section(report, "Key results")
    if key:
        lines += [key, ""]
    lines += _equation_lines(getattr(outcome, "equation", None))
    lines += _check_lines(outcome.gate)
    lines += _review_lines(outcome.verdict)
    lines.append(FULL_REPORT_HINT)
    return "\n".join(lines).strip() + "\n"
