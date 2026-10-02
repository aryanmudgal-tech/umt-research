# Brief: 25 m bridge beam

Analyze the professor's bridge beam:

- Simply supported: pinned at one end, roller at the other.
- Span: 25 m.
- Loading: uniform distributed load q = 30 kN/m acting downward over the full span.

Report the midspan deflection, the maximum bending moment, and the support shear forces.

Using this equation:
{
  "_comment": "A beam resting on soil: the ground pushes back, which adds a k*v term (the Winkler foundation) to the bending equation.",
  "label": "Beam on elastic foundation",
  "coeffs": {
    "v4": "E*I",
    "v2": "0",
    "v1": "0",
    "v0": "k"
  },
  "rhs": "q",
  "params": {
    "E": 30e9,
    "I": 0.005,
    "k": 1.0e7,
    "q": -30e3
  }
}
