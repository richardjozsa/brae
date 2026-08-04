# Delta wing at 30° angle of attack — SA-IDDES on one GPU

Leading-edge vortex system over a slender delta wing, run end to end with
`brae_pimpleFoam`: snappyHexMesh → 3.79M cells → 0.2 s of physical time on a
single GB10 in **63 minutes**.

|                  |                                                          |
|------------------|----------------------------------------------------------|
| Cells            | 3,789,669                                                |
| Turbulence       | Spalart-Allmaras IDDES, `IDDESDelta`                     |
| Wall treatment   | `nutUSpaldingWallFunction`, **no prism layers**           |
| Freestream       | 50 m/s at +30°, ν = 1e-4 → Re_MAC ≈ 1.7×10⁶              |
| Time step        | 1e-4 s fixed, 2000 steps to t = 0.2 s                    |
| Hardware         | 1× GB10, 63 min wall-clock                               |

This is a **capability demo, not a validated delta-wing study**. There is no
experimental comparison and no force integration — the interesting output is the
vortex structure, not a number. A delta wing is a good demo case precisely
because the physics is dominated by two large, well-organised leading-edge
vortices that any reasonable DES will produce; reproducing them is not evidence
of quantitative accuracy.

## The farfield boundary condition matters more than it looks

The `farfield` patch uses `freestreamVelocity` / `freestreamPressure`, **not**
`slip`. This is not a stylistic choice.

`slip` enforces U·n = 0 at the boundary. On a farfield box whose faces are not
aligned with the freestream, that constraint *rotates the oncoming flow to be
tangent to the box*, and the wing ends up seeing an angle of attack that is not
the one you set. An earlier version of this case used `slip` and, at a nominal
+30°, the wing actually saw roughly −15°. It converged cleanly and produced
plausible-looking vortices the whole time.

`freestreamVelocity` is a mixed (Robin) condition that switches between inlet
and outlet behaviour per face according to the local flow direction, so the
far boundary does not steer the flow. If you adapt this case, keep it.

## You must supply the geometry

`constant/triSurface/deltawing.stl` is **not shipped** — the surface used for the
published video came from a public model source with unclear licence terms, so
redistributing it here would be wrong. Any slender delta planform will do; the
dictionaries do not care which.

Two things to check on whatever you use:

- **Watertightness.** `surfaceCheck` must report a closed surface. snappyHexMesh
  will happily build a leaking mesh from an open STL without complaining loudly.
- **Placement and scale.** `system/snappyHexMeshDict` assumes the wing spans
  roughly x[0, 1.9], y[−0.72, 0.72], z[−0.06, 0.85] — that is what the
  `vortexBox` refinement region is sized around. The background block is
  x[−3, 9], y[−4, 4], z[−4, 4]. Rescale the box, or the wing, to match.

## Running it

```bash
cp -r 0.orig 0
blockMesh
surfaceFeatureExtract
snappyHexMesh -overwrite
brae_pimpleFoam
```

`snappyHexMesh` is the only CPU step. The refinement is driven by the
`vortexBox` region (level 4) rather than by surface refinement alone, because
the point of the case is the vortex *wake*, not the surface pressure — most of
the cell budget deliberately sits in the volume above and behind the wing where
the vortices live.

## Rendering

The published video is a Q-criterion isosurface coloured by velocity magnitude.
Any ParaView/pyvista pipeline that reads the time series will do; the one used
for the F-16 demo (`demo/f16/render.py`) works here unchanged apart from the
case path and the camera framing.
