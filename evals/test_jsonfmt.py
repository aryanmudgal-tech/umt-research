"""Tests for the display formatter behind both trace renderers.

Pure functions over plain data: no API key, no network, no files.
"""

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent.jsonfmt import compact_json, was_folded

SUPPORTS = {
    "N1": [True, True, True, True, False, False],
    "N5": [False, True, True, False, False, False],
}

MODEL = {
    "nodes": [{"id": f"N{k + 1}", "x": 6.25 * k} for k in range(5)],
    "supports": SUPPORTS,
    "material": {"E": 30000000000.0, "G": 12500000000.0},
    "section": {"A": 0.5, "Iy": 0.005, "Iz": 0.005, "J": 0.001},
    "distributed_loads": [
        {"element": "all", "direction": "y", "w1": -30000.0, "w2": -30000.0}
    ],
    "point_loads": [],
}


def lines(obj, **kwargs):
    return compact_json(obj, **kwargs).split("\n")


def test_primitive_array_goes_inline():
    out = compact_json({"N1": [True, True, True, True, False, False]})

    assert out == '{"N1": [true, true, true, true, false, false]}'
    assert "\n" not in out  # the twelve-line support vector is gone


def test_every_primitive_kind_inlines():
    out = compact_json({"k": [1, -2.5, "s", True, False, None]})

    assert out == '{"k": [1, -2.5, "s", true, false, null]}'


def test_long_primitive_array_wraps_but_never_one_per_line():
    body = lines({"xs": list(range(60))})

    assert len(body) < 12, body  # 60 numbers, not 60 lines
    packed = [line for line in body if line.strip().startswith(("0", "1", "2", "3"))]
    assert packed and all(line.count(",") >= 2 for line in packed)
    for line in body:
        assert len(line) <= 88


def test_array_of_objects_folds_with_an_accurate_count():
    nodes = [{"id": f"N{n}", "x": float(n)} for n in range(9)]
    body = lines({"nodes": nodes})

    assert body[1].strip() == '"nodes": ['
    assert body[2].strip() == '{"id": "N0", "x": 0.0},'
    assert body[5].strip() == '{"id": "N3", "x": 3.0},'
    assert body[6].strip() == "... 5 more items (9 total)"
    assert "N8" not in "\n".join(body)  # folded, and the marker says so


def test_arrays_barely_past_the_limit_are_shown_whole():
    """The marker costs a line, so hiding one or two items buys nothing."""
    out = compact_json({"nodes": MODEL["nodes"]})  # 5 nodes, fold_after=4

    assert not was_folded(out)
    assert "N5" in out


def test_folding_never_hides_fewer_than_three():
    """Why the marker is never singular: the margin guarantees it."""
    for total in range(1, 40):
        for limit in range(0, 12):
            out = compact_json(
                {"pts": [{"i": n, "pad": "x" * 30} for n in range(total)]},
                fold_after=limit,
            )
            hidden = re.search(r"\.\.\. (\d+) more items", out)
            assert hidden is None or int(hidden.group(1)) >= 3


def test_fold_count_matches_what_was_dropped():
    items = [{"i": n, "keep": False} for n in range(30)]
    out = compact_json({"items": items}, fold_after=6)

    assert "... 24 more items (30 total)" in out
    assert out.count('"i":') == 6


def test_a_line_that_already_fits_is_never_folded():
    """Folding stops a block exploding; one line cannot explode."""
    out = compact_json({"pts": [{"i": n} for n in range(6)]})

    assert not was_folded(out)  # six items, fold_after four, still one line
    assert out.count('"i"') == 6
    assert len(out.split("\n")) == 1


def test_fold_after_zero_keeps_only_the_marker():
    out = compact_json({"pts": [{"i": n, "pad": "x" * 30} for n in range(3)]},
                       fold_after=0)

    assert "... 3 more items (3 total)" in out
    assert '"i"' not in out


def test_short_object_arrays_are_not_folded():
    out = compact_json({"loads": MODEL["distributed_loads"]})

    assert not was_folded(out)
    assert "-30000.0" in out


def test_small_objects_render_inline():
    body = lines(MODEL["section"])

    assert body == ['{"A": 0.5, "Iy": 0.005, "Iz": 0.005, "J": 0.001}']


def test_object_too_wide_to_inline_keeps_one_key_per_line():
    wide = {"description": "x" * 120, "second": 2}
    body = lines(wide)

    assert body[0] == "{"
    assert body[1].startswith('  "description": "xxx')
    assert body[2].strip() == '"second": 2'
    assert body[3] == "}"


def test_nested_depth_is_preserved():
    body = lines({"a": {"b": {"c": {"d": ["x" * 40, "y" * 40, "z" * 40]}}}})

    assert body[0] == "{"
    assert body[1] == '  "a": {'
    assert body[2] == '    "b": {'
    assert body[3] == '      "c": {'
    assert body[4] == '        "d": ['
    assert body[5] == '          "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",'
    assert body[-5:] == ["        ]", "      }", "    }", "  }", "}"]


def test_indent_width_is_honoured():
    body = lines({"a": {"description": "x" * 120, "second": 2}}, indent=4)

    assert body[1] == '    "a": {'
    assert body[2].startswith('        "description"')


def test_inline_width_decides_what_fits():
    obj = {"section": {"A": 0.5, "Iy": 0.005, "Iz": 0.005, "J": 0.001}}

    assert len(lines(obj, inline_width=200)) == 1
    assert len(lines(obj, inline_width=20)) > 1


def test_empty_containers():
    assert compact_json({}) == "{}"
    assert compact_json([]) == "[]"
    assert compact_json({"point_loads": [], "meta": {}}) == '{"point_loads": [], "meta": {}}'
    assert compact_json({"a": [[], {}]}) == '{"a": [[], {}]}'


def test_bare_primitives():
    assert compact_json(None) == "null"
    assert compact_json(True) == "true"
    assert compact_json(3.5) == "3.5"
    assert compact_json("hi") == '"hi"'


def _diagrams(count):
    return [
        {"x": 0.5 * n, "N": -0.0, "Vy": 375000.0, "Vz": 0.0, "T": -0.0, "My": -0.0,
         "Mz": 445312.5}
        for n in range(count)
    ]


def test_a_huge_diagrams_array_stays_short():
    body = lines({"diagrams": _diagrams(500)})

    assert len(body) < 45, len(body)
    assert "... 496 more items (500 total)" in "\n".join(body)
    assert was_folded("\n".join(body))


def test_output_length_does_not_grow_with_the_array():
    """The whole point: 24 sample points and 5000 cost the same height."""
    assert len(lines({"diagrams": _diagrams(24)})) == len(lines({"diagrams": _diagrams(5000)}))


def test_the_whole_model_dict_reads_at_a_glance():
    body = lines({"model": MODEL})
    text = "\n".join(body)

    assert len(body) < 40, len(body)
    for key in MODEL:
        assert f'"{key}"' in text, key
    assert '"N1": [true, true, true, true, false, false]' in text
    assert "30000000000.0" in text and "12500000000.0" in text
    assert "-30000.0" in text


def test_key_order_is_preserved():
    out = compact_json({"z": 1, "a": 2, "m": 3})

    assert out == '{"z": 1, "a": 2, "m": 3}'


def test_unicode_survives_and_quotes_are_escaped():
    out = compact_json({"note": "béton armé — 25 m", "quoted": 'he said "no"'})

    assert "béton armé — 25 m" in out
    assert "\\u" not in out
    assert '\\"no\\"' in out


def test_control_characters_never_add_lines():
    out = compact_json({"text": "one\ntwo\tthree", "path": "C:\\tmp"})

    assert "\n" not in out  # the newline is escaped, not rendered
    assert "\\n" in out and "\\t" in out and "\\\\tmp" in out


def test_non_serialisable_values_become_strings():
    out = compact_json({"obj": object(), "set": {1}})

    assert "<object object at" in out
    assert out.startswith("{") and out.endswith("}")


def test_integer_keys_become_strings():
    assert compact_json({1: "a", 2: "b"}) == '{"1": "a", "2": "b"}'


def test_tuples_render_like_arrays():
    assert compact_json({"k": (1, 2, 3)}) == '{"k": [1, 2, 3]}'


def test_was_folded_only_fires_on_the_marker():
    assert not was_folded("")
    assert not was_folded(compact_json(MODEL["section"]))
    assert not was_folded('{"note": "... 3 more items (9 total) was the old text"}')
    assert was_folded(compact_json({"nodes": MODEL["nodes"]}, fold_after=1))


def test_unfolded_output_still_parses_as_json():
    obj = {"model": {k: v for k, v in MODEL.items() if k != "nodes"}}
    out = compact_json(obj)

    assert not was_folded(out)
    assert json.loads(out) == obj


def test_output_has_no_trailing_newline():
    out = compact_json(MODEL)

    assert not out.endswith("\n")
    assert not out.startswith("\n")
