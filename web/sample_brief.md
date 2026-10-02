# Brief: 25 m beam on soil

Analyze a bridge beam that rests on the ground along its whole length:

- Simply supported: pinned at one end, roller at the other.
- Span: 25 m.
- Elastic modulus: E = 30 GPa.
- Second moment of area about the bending axis: I = 0.005 m^4.
- Loading: uniform distributed load q = 30 kN/m acting downward over the full span.
- The beam rests on soil with a modulus of subgrade reaction k = 1.0e7 N/m per metre of span.

Report the midspan deflection, the maximum bending moment, and the support
shear forces.

Use this governing equation (EI v'''' + k v = q, with downward negative):

```json
{
  "label": "Beam on elastic foundation",
  "coeffs": {"v4": "E*I", "v2": "0", "v1": "0", "v0": "k"},
  "rhs": "q",
  "params": {"E": 30e9, "I": 0.005, "k": 1.0e7, "q": -30e3}
}
```

To analyze a different beam, change the numbers above. To change the physics,
describe it in words (an axial force, a depth that varies along the span) or
edit the equation: v4 is the bending stiffness, v2 an axial force
(compression positive), v0 the soil stiffness, and rhs the load.
