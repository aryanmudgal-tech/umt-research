# Brief: 25 m beam on soil

A bridge beam resting on the ground along its whole length.

- Simply supported: pinned at one end, roller at the other.
- Span: 25 m.
- Elastic modulus: E = 30 GPa.
- Second moment of area about the bending axis: I = 0.005 m⁴.
- Loading: uniform distributed load q = 30 kN/m, downward, over the full span.

## Governing equation

Beam on elastic foundation (Winkler):

$$EI\,\frac{d^4v}{dx^4} + k\,v = q$$

- E = 30 GPa, I = 0.005 m⁴
- k = 1.0 × 10⁷ N/m² (modulus of subgrade reaction, per metre of beam)
- q = 30 kN/m, downward

## Report

The midspan deflection, the maximum bending moment, and the support shear
forces.

---

To analyze another beam, change the numbers above. To change the physics,
change the equation: an axial force adds a P·d²v/dx² term (compression
positive), and a section that varies along the span makes EI a function of x.
