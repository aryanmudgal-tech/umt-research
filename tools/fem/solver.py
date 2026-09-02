"""3D Euler-Bernoulli beam FEM solver (Galerkin / Hermite, 12-DOF elements).

Beam axis is global X, so local and global axes coincide and no rotation
transformation is needed. DOF order per node: [ux, uy, uz, rx, ry, rz].
Sign convention: theta_y = -d(uz)/dx, which makes the x-z bending block the
standard Hermite block with the rotation rows/columns negated.

Units: SI throughout (m, N, Pa).
"""

import math

import numpy as np

_DOF_NAMES = ("FX", "FY", "FZ", "MX", "MY", "MZ")
_DOF_INDEX = {name: k for k, name in enumerate(_DOF_NAMES)}

# element-local DOF slots for each stiffness block
_BEND_Y_PLANE = [1, 5, 7, 11]  # uy_i, rz_i, uy_j, rz_j  (bending about z)
_BEND_Z_PLANE = [2, 4, 8, 10]  # uz_i, ry_i, uz_j, ry_j  (bending about y)


def _elements(model):
    """Element list, defaulting to consecutive-node connectivity."""
    elems = model.get("elements")
    if elems:
        return elems
    nodes = model["nodes"]
    return [
        {"id": f"E{k + 1}", "i": nodes[k]["id"], "j": nodes[k + 1]["id"]}
        for k in range(len(nodes) - 1)
    ]


def _hermite_block(EI, L, sign):
    """4x4 bending block; sign=-1 applies the theta_y = -d(uz)/dx convention."""
    s = sign
    return (
        EI
        / L**3
        * np.array(
            [
                [12.0, s * 6 * L, -12.0, s * 6 * L],
                [s * 6 * L, 4 * L**2, -s * 6 * L, 2 * L**2],
                [-12.0, -s * 6 * L, 12.0, -s * 6 * L],
                [s * 6 * L, 2 * L**2, -s * 6 * L, 4 * L**2],
            ]
        )
    )


def _k_element(E, G, A, Iy, Iz, J, L):
    """12x12 element stiffness (Logan / Przemieniecki), local == global axes."""
    k = np.zeros((12, 12))
    for a, b, c in ((0, 6, E * A / L), (3, 9, G * J / L)):
        k[a, a] = k[b, b] = c
        k[a, b] = k[b, a] = -c
    k[np.ix_(_BEND_Y_PLANE, _BEND_Y_PLANE)] += _hermite_block(E * Iz, L, +1)
    k[np.ix_(_BEND_Z_PLANE, _BEND_Z_PLANE)] += _hermite_block(E * Iy, L, -1)
    return k


def _f_equivalent(direction, w1, w2, L):
    """Consistent nodal loads for a linear distributed load w1@i -> w2@j."""
    F1 = L * (7 * w1 + 3 * w2) / 20
    F2 = L * (3 * w1 + 7 * w2) / 20
    M1 = L**2 * (3 * w1 + 2 * w2) / 60
    M2 = L**2 * (2 * w1 + 3 * w2) / 60
    f = np.zeros(12)
    if direction == "y":
        f[_BEND_Y_PLANE] = [F1, M1, F2, -M2]
    elif direction == "z":
        f[_BEND_Z_PLANE] = [F1, -M1, F2, M2]  # moment signs follow theta_y = -uz'
    else:
        raise ValueError(f"distributed load direction must be 'y' or 'z', got {direction!r}")
    return f


def _element_line_loads(model, elems, x_of):
    """Resolve distributed loads to per-element {elem_id: {'y': (w1, w2), 'z': ...}}.

    An 'all' load is interpolated linearly over the whole beam (w1 at the first
    node, w2 at the last), so a varying load stays continuous across elements.
    """
    loads = {e["id"]: {"y": [0.0, 0.0], "z": [0.0, 0.0]} for e in elems}
    dist = model.get("distributed_loads") or []
    if not dist:
        return loads
    x0 = min(x_of.values())
    span = max(x_of.values()) - x0
    for dl in dist:
        d = dl["direction"]
        if d not in ("y", "z"):
            raise ValueError(f"distributed load direction must be 'y' or 'z', got {d!r}")
        targets = elems if dl["element"] == "all" else [e for e in elems if e["id"] == dl["element"]]
        if not targets:
            raise ValueError(f"distributed load references unknown element {dl['element']!r}")
        for e in targets:
            if dl["element"] == "all" and span > 0:
                wi = dl["w1"] + (dl["w2"] - dl["w1"]) * (x_of[e["i"]] - x0) / span
                wj = dl["w1"] + (dl["w2"] - dl["w1"]) * (x_of[e["j"]] - x0) / span
            else:
                wi, wj = dl["w1"], dl["w2"]
            loads[e["id"]][d][0] += wi
            loads[e["id"]][d][1] += wj
    return loads


def assemble(model):
    """Assemble the global system.

    Returns (K, F, dof_map): K = global stiffness BEFORE supports are applied,
    F = global load vector (point loads + consistent distributed loads),
    dof_map = {(node_id, dof_index): global_index}.
    """
    nodes = model["nodes"]
    x_of = {n["id"]: n["x"] for n in nodes}
    dof_map = {(n["id"], d): 6 * k + d for k, n in enumerate(nodes) for d in range(6)}

    n_dof = 6 * len(nodes)
    K = np.zeros((n_dof, n_dof))
    F = np.zeros(n_dof)

    mat, sec = model["material"], model["section"]
    elems = _elements(model)
    line_loads = _element_line_loads(model, elems, x_of)

    for e in elems:
        L = x_of[e["j"]] - x_of[e["i"]]
        if L <= 0:
            raise ValueError(f"element {e['id']!r} has non-positive length {L}")
        k_e = _k_element(mat["E"], mat["G"], sec["A"], sec["Iy"], sec["Iz"], sec["J"], L)
        f_e = np.zeros(12)
        for d in ("y", "z"):
            w1, w2 = line_loads[e["id"]][d]
            if w1 or w2:
                f_e += _f_equivalent(d, w1, w2, L)
        idx = [dof_map[(e["i"], d)] for d in range(6)] + [dof_map[(e["j"], d)] for d in range(6)]
        K[np.ix_(idx, idx)] += k_e
        F[idx] += f_e

    for pl in model.get("point_loads") or []:
        F[dof_map[(pl["node"], _DOF_INDEX[pl["dof"]])]] += pl["value"]

    return K, F, dof_map


def solve_beam_3d(model):
    """Solve the beam model; see module docstring for conventions.

    Returns the result dict: displacements, reactions at restrained DOFs,
    internal-force diagrams sampled along the beam, max absolute values,
    and the total applied load.
    """
    K, F, dof_map = assemble(model)
    n_dof = K.shape[0]

    restrained = sorted(
        dof_map[(node, d)]
        for node, flags in model["supports"].items()
        for d in range(6)
        if flags[d]
    )
    free = [g for g in range(n_dof) if g not in set(restrained)]

    u = np.zeros(n_dof)
    try:
        u[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])
    except np.linalg.LinAlgError as exc:
        raise ValueError("structure is unstable: singular stiffness matrix") from exc

    # reactions recovered exactly from the full (unmodified) system
    R = K @ u - F

    displacements = {
        n["id"]: {
            name: u[dof_map[(n["id"], d)]]
            for d, name in enumerate(("ux", "uy", "uz", "rx", "ry", "rz"))
        }
        for n in model["nodes"]
    }
    reactions = {
        node: {_DOF_NAMES[d]: R[dof_map[(node, d)]] for d in range(6) if flags[d]}
        for node, flags in model["supports"].items()
        if any(flags)
    }

    diagrams = _diagrams(model, u, dof_map)

    max_abs = {
        "uy": max(abs(v["uy"]) for v in displacements.values()),
        "uz": max(abs(v["uz"]) for v in displacements.values()),
    }
    for key in ("Mz", "My", "Vy", "Vz", "N", "T"):
        max_abs[key] = max(abs(p[key]) for p in diagrams)

    total = _total_applied(model)

    return {
        "displacements": displacements,
        "reactions": reactions,
        "diagrams": diagrams,
        "max_abs": max_abs,
        "total_applied": total,
    }


def _diagrams(model, u, dof_map):
    """Sample internal forces along each element from end forces + element loads.

    Section forces come from equilibrium of the element segment left of the cut:
    N and T positive in tension / positive twist, Vy = sum of +y forces on the
    left segment (so dMz/dx = Vy and sagging Mz is positive under -y load).
    """
    nodes = model["nodes"]
    x_of = {n["id"]: n["x"] for n in nodes}
    mat, sec = model["material"], model["section"]
    elems = sorted(_elements(model), key=lambda e: x_of[e["i"]])
    line_loads = _element_line_loads(model, elems, x_of)

    n_per = max(3, math.ceil(20 / len(elems)) + 1)  # >= 21 samples in total
    points = []
    for e in elems:
        L = x_of[e["j"]] - x_of[e["i"]]
        k_e = _k_element(mat["E"], mat["G"], sec["A"], sec["Iy"], sec["Iz"], sec["J"], L)
        f_eq = np.zeros(12)
        (wy1, wy2), (wz1, wz2) = line_loads[e["id"]]["y"], line_loads[e["id"]]["z"]
        if wy1 or wy2:
            f_eq += _f_equivalent("y", wy1, wy2, L)
        if wz1 or wz2:
            f_eq += _f_equivalent("z", wz1, wz2, L)
        idx = [dof_map[(e["i"], d)] for d in range(6)] + [dof_map[(e["j"], d)] for d in range(6)]
        f_end = k_e @ u[idx] - f_eq  # nodal forces acting ON the element

        by, bz = (wy2 - wy1) / L, (wz2 - wz1) / L
        for s in np.linspace(0.0, L, n_per):
            Wy = wy1 * s + by * s**2 / 2  # resultant of w_y on [0, s]
            Wz = wz1 * s + bz * s**2 / 2
            My_wy = wy1 * s**2 / 2 + by * s**3 / 6  # its moment about the cut
            My_wz = wz1 * s**2 / 2 + bz * s**3 / 6
            points.append(
                {
                    "x": x_of[e["i"]] + s,
                    "N": -f_end[0],
                    "Vy": f_end[1] + Wy,
                    "Vz": f_end[2] + Wz,
                    "T": -f_end[3],
                    "My": -f_end[4] - f_end[2] * s - My_wz,
                    "Mz": -f_end[5] + f_end[1] * s + My_wy,
                }
            )
    return points


def _total_applied(model):
    """Sum of all applied loads, distributed loads integrated over their span."""
    total = {"FX": 0.0, "FY": 0.0, "FZ": 0.0}
    for pl in model.get("point_loads") or []:
        if pl["dof"] in total:
            total[pl["dof"]] += pl["value"]
    x_of = {n["id"]: n["x"] for n in model["nodes"]}
    elems = _elements(model)
    line_loads = _element_line_loads(model, elems, x_of)
    for e in elems:
        L = x_of[e["j"]] - x_of[e["i"]]
        (wy1, wy2), (wz1, wz2) = line_loads[e["id"]]["y"], line_loads[e["id"]]["z"]
        total["FY"] += (wy1 + wy2) / 2 * L
        total["FZ"] += (wz1 + wz2) / 2 * L
    return total
