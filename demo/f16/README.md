# F-16 at 32° angle of attack — SA-IDDES on one GPU

A full-aircraft detached-eddy simulation run end to end with `brae_pimpleFoam`:
snappyHexMesh → 25.2M cells → 1.5 s of physical time on a single H100 in 4.4 h.

|                  |                                                         |
|------------------|---------------------------------------------------------|
| Cells            | 25,233,348 (76.1M faces)                                 |
| Turbulence       | Spalart-Allmaras IDDES, `IDDESDelta`                     |
| Wall treatment   | `nutUSpaldingWallFunction`, **no prism layers** (y+ ≈ 200) |
| Freestream       | 50 m/s at +32°, ν = 1e-4 → Re_MAC ≈ 1.7×10⁶              |
| Time step        | 3e-4 s fixed; CFL mean 0.28, p99 0.40, max 1.48          |
| Peak VRAM        | 43.9 GB of 80 (~1.8 kB/cell)                             |

This is a **capability demo, not a validated F-16 study**. No experimental
comparison, no force coefficients, and the wall treatment is modelled rather
than resolved — do not read C_L/C_D off it.

## You must supply the geometry

`constant/triSurface/f16.stl` is **not shipped**. The model used to produce the
published videos was downloaded from a public model site with unclear licence
terms, so redistributing it here would be wrong. Drop any F-16 surface in as
`f16.stl`; the dictionaries do not care which.

Two things to check on whatever you use, because they bit us:

- **Watertightness.** `surfaceCheck` must report a closed surface. Print-oriented
  STLs frequently are not, and snappy will produce a leaking mesh without saying
  so loudly.
- **Dimensions.** Ours measured 15.00 m long but only 8.99 m span and 4.23 m tall
  against a real 15.06 / 9.45 / 5.09 — about 5% narrow and 17% short, which made
  the nose visibly too slender. Fine for a flow demo, wrong for anything quantitative.

`system/snappyHexMeshDict` assumes the aircraft is centred near the origin,
nose toward −x, spanning roughly x[−7.5, 7.5].

## Running it

```bash
blockMesh
surfaceFeatureExtract
decomposePar && mpirun -np 20 snappyHexMesh -overwrite -parallel && reconstructParMesh -constant
rm -rf processor* && cp -r 0.orig 0
brae_pimpleFoam
```

Meshing takes ~8 min on 20 cores. The solve is ~3.3 s/step at 25M cells on an
H100; `endTime 1.5` is 5000 steps.

## The boundary condition that matters

The far field is `freestreamVelocity` / `freestreamPressure`, **not `slip`**.

This is not a stylistic choice. `slip` forces `U·n = 0` at the boundary, so the
top and bottom planes cannot pass the vertical mass flux that an angled inflow
injects — the boundary quietly turns the flow and cancels the angle of attack.
Our first run of this case used `slip` and produced an aircraft effectively at
**−15°** (air arriving from *above*) while the case file said +32°. Residuals
were clean, continuity was 1e-10, the solver exited 0, and the result was
physically meaningless.

Verify it rather than trusting the dictionary. Sample U in a box well upstream
and check the angle:

```python
ang = np.degrees(np.arctan2(U[:,2], U[:,0]))   # want +32, not -15
```

This run measures **+32.45°** upstream at t=1.5. Outboard reads +36° and above
the aircraft +27° — that is real upwash and downwash from a lifting body, not error.

## Rendering

`render.py` produces three camera views (orbiting, fixed top, fixed corner) from
a single pass, computing Q-criterion once per timestep and drawing each view from
it — three videos for ~1.7× the cost of one.

```bash
CASE=$PWD PATCH=f16 BOX='[-10,13,-7,7,-5,7]' CELL_L_MAX=0.06 \
QLEV=150,1.5e3,1.5e4 LABEL="F-16 32deg AoA" python3 render.py
```

Pick `QLEV` from your own field, not from this example. Q scales roughly as
1/Δx², so it shifts by orders of magnitude with mesh and flow speed — at Q=800
this same field showed a handful of scattered specks; at Q=150 it showed the
entire separated wake.
