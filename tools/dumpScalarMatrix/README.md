# dumpScalarMatrix — an OpenFOAM utility for comparing matrix assembly

Built to answer one question directly instead of by inference: **does brae assemble the same matrix as
OpenFOAM?** OpenFOAM offers no way to inspect an `fvScalarMatrix` from outside a solver, so every earlier
attempt compared *fields* and reasoned backwards about the operator. This dumps the matrix itself.

## Build

```sh
source /usr/lib/openfoam/openfoam2412/etc/bashrc
cd tools/dumpScalarMatrix && wmake
```

It links the turbulence libraries so wall-function boundary types (`omegaWallFunction`, `kqRWallFunction`)
can be *constructed* when reading fields. Note they still cannot be *evaluated* here: they look up a
registered `turbulenceModel`, which a standalone utility has none of. Use plain BC types in the comparison
case (see below).

## Use

```sh
dumpScalarMatrix -case <dir> -field omega -gamma 1e-3     # writes Ddiag, Dsource in the latest time
./build/diag_compare <dir> <time> 1e-3                    # brae's side, cell by cell
```

It assembles `fvm::div(phi, psi) - fvm::laplacian(gamma, psi)` with `phi = fvc::flux(U)` and a uniform
diffusivity, honouring the case's own `fvSchemes` — so `bounded Gauss linearUpwind grad(omega)` is
exercised. It is deliberately **not** the full kOmegaSST omega equation: `F1`/`beta`/`gamma`/`CDkOmega`
are protected members of `kOmegaSSTBase`, so rebuilding them here would test a reimplementation rather
than OpenFOAM itself. The machinery under investigation — the bounded prefix, the upwind matrix, the
deferred correction, the laplacian — is generic, and that is what this builds.

## Result it produced

On pitzDaily kOmegaSST at a converged state, comparing against `diag_compare`:

| quantity | brae vs OF |
|---|---|
| matrix diagonal | 1.76e-06 |
| source (deferred correction) | 6.66e-07 |

Both at the ASCII write-precision floor: **brae's scalar-transport assembly is identical to OpenFOAM's.**
That closed the "brae's omega diagonal is weaker, so the explicit correction over-amplifies" hypothesis —
the diagonal is not weaker, and the correction is not larger.

## A harness trap worth remembering

The first run reported a diagonal difference of 7.7e-04 with **100% of the error energy in boundary
cells**. That was the harness, not the code: the comparison had forced `zeroGradient` on every U patch,
while OpenFOAM read U's real BCs (fixedValue inlet, noSlip walls), so `phi = fvc::flux(U)` differed on the
boundary faces. Honouring U's actual BCs dropped the difference to 1.76e-06. If this utility ever reports
a boundary-localised difference, suspect the field setup before the solver.
