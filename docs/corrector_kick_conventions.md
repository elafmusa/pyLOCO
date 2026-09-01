# Corrector kick conventions

`RMConfig.dkick` is an integrated physical angular kick in radians. A bipolar
ORM column is the orbit difference between `+dkick/2` and `-dkick/2`; an
unnormalized column therefore represents the orbit change for the full
integrated `dkick`.

- **Linear:** inserts `dkick` directly into the canonical momenta (`px`, `py`)
  and propagates it through the element and one-turn maps. It does not interpret
  the selected element's stored corrector coefficient.
- **Analytical:** multiplies the same integrated `dkick` by the closed-orbit
  response formula. Its direct-plane signs follow the Linear convention.
- **Tracking:** perturbs the selected AT element and subtracts closed orbits.
  `KickAngle` is already integrated. For finite multipoles AT uses
  `dpx=-Length*dPolynomB[0]` and `dpy=+Length*dPolynomA[0]`; thin-multipole
  polynomial coefficients are integrated and use the same signs. A zero-length
  thick-pass multipole cannot represent a nonzero kick through its polynomial.
- **pySC measurement:** public corrector setpoints such as `B1L` and `A1L` are
  integrated strengths. pySC divides them by a finite magnet's length before
  updating the underlying AT polynomial. The raw orbit then follows AT's
  horizontal/vertical polynomial signs; a machine driver may additionally map
  hardware polarity to the pyLOCO convention. The PETRA-IV campaign does this
  explicitly by flipping its selected horizontal ORM columns. Its bipolar,
  unnormalized column still represents the full integrated `dkick`.

The conversion is determined from each element's representation and length;
there is no lattice-specific length or polarity constant.
