"""Independent cross-check: solve the shared model dict with PyNiteFEA.

Builds the exact same model in PyNite 3.0 so results from tools.fem.solver
can be compared against a second, independently written FEM engine.
PyNite's signed member dist load in 'FY'/'FZ' matches the model dict's
convention (negative w = downward), verified against the SS-UDL closed form.
"""

import numpy as np
from Pynite import FEModel3D

_DISP_KEYS = {
    "ux": "DX", "uy": "DY", "uz": "DZ",
    "rx": "RX", "ry": "RY", "rz": "RZ",
}
_RXN_KEYS = ["FX", "FY", "FZ", "MX", "MY", "MZ"]

# Points per member for PyNite's own deflected shape. The comparison it
# feeds is quoted to a few significant figures, and a smooth field is flat at
# its peak, so 101 points put the height within about 1e-4 of the member's own
# extremum - far below the difference between two engines' interpolations.
_MEMBER_POINTS = 101


def _elements_of(model):
    """Element list, defaulting to consecutive-node elements.

    Default ids must be E1..En exactly as tools.fem.solver generates them, or
    element-targeted loads would land on different elements in the two engines.
    """
    if model.get("elements"):
        return model["elements"]
    nodes = model["nodes"]
    return [
        {"id": f"E{k + 1}", "i": nodes[k]["id"], "j": nodes[k + 1]["id"]}
        for k in range(len(nodes) - 1)
    ]


def solve_with_pynite(model: dict) -> dict:
    m = FEModel3D()

    E = model["material"]["E"]
    G = model["material"]["G"]
    nu = E / (2.0 * G) - 1.0  # consistent with E, G; does not affect beam stiffness
    m.add_material("mat", E, G, nu, 0.0)

    s = model["section"]
    m.add_section("sec", A=s["A"], Iy=s["Iy"], Iz=s["Iz"], J=s["J"])

    for n in model["nodes"]:
        m.add_node(n["id"], n["x"], 0.0, 0.0)

    elements = _elements_of(model)
    for e in elements:
        m.add_member(e["id"], e["i"], e["j"], "mat", "sec")

    for node_id, restrained in model.get("supports", {}).items():
        m.def_support(node_id, *restrained)

    for dl in model.get("distributed_loads", []):
        direction = "FY" if dl["direction"] == "y" else "FZ"
        targets = elements if dl["element"] == "all" else [
            e for e in elements if e["id"] == dl["element"]
        ]
        for e in targets:
            m.add_member_dist_load(e["id"], direction, dl["w1"], dl["w2"])

    for pl in model.get("point_loads", []):
        m.add_node_load(pl["node"], pl["dof"], pl["value"])

    m.analyze()

    displacements = {}
    reactions = {}
    for n in model["nodes"]:
        node = m.nodes[n["id"]]
        displacements[n["id"]] = {
            k: getattr(node, attr)["Combo 1"] for k, attr in _DISP_KEYS.items()
        }
        restrained = model.get("supports", {}).get(n["id"])
        if restrained and any(restrained):
            reactions[n["id"]] = {
                key: getattr(node, f"Rxn{key}")["Combo 1"]
                for key, flag in zip(_RXN_KEYS, restrained)
                if flag
            }

    return {
        "displacements": displacements,
        "reactions": reactions,
        "max_abs": _member_peaks(m),
    }


def _member_peaks(m) -> dict:
    """PyNite's own largest |uy| and |uz| ANYWHERE along the members.

    tools.fem.solver reports the extremum of the whole deflected shape, so the
    figure standing beside it must be the same quantity: PyNite's largest NODAL
    deflection is a different number whenever the peak falls between two nodes,
    and putting it in the same row would read as two engines disagreeing when
    they are answering two different questions. The two interpolations are not
    identical - PyNite adds the exact member-load shape where this solver
    reports its Hermite cubic - so the row still compares two independent
    engines and not one engine with itself.
    """
    peaks = {}
    for key, direction in (("uy", "dy"), ("uz", "dz")):
        peaks[key] = max(
            (
                # one sweep of the member, not the two that max_deflection and
                # min_deflection would each walk separately. Row 0 of what
                # comes back is the STATION along the member, not a deflection
                float(np.max(np.abs(member.deflection_array(direction, _MEMBER_POINTS)[1])))
                for member in m.members.values()
            ),
            default=0.0,
        )
    return peaks
