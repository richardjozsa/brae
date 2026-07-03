# Roadmap

Brae runs steady, incompressible, single-region flow on a single GPU today. Coming next:

- More steady single-phase solvers
- Multiphase solvers
- Multi-region
- Heat transfer
- Multi-GPU support

## Accuracy notes

- Matches OpenFOAM to under 1% on the fields for the validated cases
- **Near-wall (low y+):** near-wall turbulence quantities can differ ~10-15% on flat plates; bulk fields and forces still track OpenFOAM
- **Extreme aspect ratio (AR ≳ 1000):** the steady solve can settle differently on sliver cells; use the same under-relaxation as OpenFOAM
- On hard steady cases (bluff-body aero), brae plateaus its residual exactly as OpenFOAM does

**Brae never guesses:** if a model, boundary condition, or scheme is not supported, it stops at start-up naming
exactly what it found, so you never get a silently wrong result.
