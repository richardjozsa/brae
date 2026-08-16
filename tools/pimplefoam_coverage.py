#!/usr/bin/env python3
"""Generate the pimpleFoam coverage manifest by DIFFING OpenFOAM's runtime selection tables
against what brae names.

WHY THIS EXISTS. Coverage was previously discovered by running tutorials: a case failed, a gap got
found, a gap got fixed. That only ever finds gaps a sampled case happens to exercise, and it cannot
tell "not implemented" from "implemented wrong" from "never reached". `turbOnFinalIterOnly` -- an
outright PIMPLE algorithm mismatch -- survived a 30-case sweep because most tutorials use
nOuterCorrectors 1, where the right and wrong cadences coincide.

The inventory here is built from the SOURCE OF TRUTH instead: OpenFOAM registers every selectable
type with a TypeName("...") in its header, and that string is exactly what a case dictionary writes.
Enumerating those gives a finite, checkable work list that exists before any case is run.

SCOPE. Only what incompressible pimpleFoam can reach: boundary conditions registered in
finiteVolume (plus the turbulence wall functions), the finiteVolume schemes, the incompressible
turbulence models, fvOptions, and the mesh-motion classes. OpenFOAM's thermophysical, regionModel,
lagrangian and multiphase trees register many more types that this solver cannot select.

Usage:  python3 tools/pimplefoam_coverage.py [--of <OpenFOAM src>] [--out docs/pimplefoam-coverage.md]
"""

import argparse
import os
import re
import subprocess
import sys

BRAE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMANDED = set()

# Types brae REFUSES on purpose, with the reason. A refusal is a supported outcome -- the port says
# what it cannot do and stops -- so these are tracked separately from silent gaps. Keep the reason
# short; the full explanation belongs at the throw site.
REFUSED = {
    "dynamicFvMesh": {
        "interfaceTrackingFvMesh": "free-surface tracking: mesh motion driven by the interface itself",
        "dynamicMotionSolverFvMeshAMI": "topology change during the run",
        "dynamicOversetFvMesh": "overset replaces the matrix addressing",
    },
    "motionSolver": {
        "sixDoFRigidBodyMotion": "rigid-body ODE + restraints",
    },
    "RASModel": {"kkLOmega": "3-equation transition model"},
    "LESModel": {"kEqn": "one-equation SGS k transport"},
    "fvPatchField": {
        "velocityFilmShell": "finite-area film region coupled to the 3-D solve",
        "turbulentDFSEMInlet": "synthetic-eddy inflow generator",
        "cyclicSlip": "coupled patch brae does not build",
    },
}

# Deliberate semantic divergences: implemented, but NOT identical to OpenFOAM, and said so at runtime.
# The policy is that any divergence is chosen and named -- never accidental.
DIVERGENT = {
    "ddtScheme": {
        "backward": "cell ddt implemented; fvc::ddtCorr flux coupling omitted (notice)",
        "CrankNicolson": "cell ddt implemented; fvc::ddtCorr flux coupling omitted (notice)",
        "bounded": "parsed as Euler, not OpenFOAM's bounded behaviour",
    },
}


def sh(cmd):
    # errors="replace": OpenFOAM tutorials ship binary and gzipped field files, and one undecodable
    # byte in a 0/ directory would otherwise abort the whole inventory.
    r = subprocess.run(cmd, shell=True, capture_output=True)
    return r.stdout.decode("utf-8", errors="replace")


def typenames(paths):
    """Every TypeName("x") under the given OpenFOAM paths -- the exact string a dict writes."""
    out = set()
    for p in paths:
        if not os.path.isdir(p):
            continue
        txt = sh(f"grep -rhoE 'TypeName\\(\"[A-Za-z0-9_]+\"\\)' {p} 2>/dev/null")
        out |= set(re.findall(r'TypeName\("([A-Za-z0-9_]+)"\)', txt))
    return out


def rts(paths, table):
    """addToRunTimeSelectionTable(<table>, X, ...) -> the X's. Used where a category's selectable
    names are NOT the directory names: the surface-interpolation limiters and the motion solvers both
    register short names from macros, and scanning directories instead pulled in base classes
    (LimitedScheme, PhiScheme) and helper classes that no dictionary can select."""
    out = set()
    for p_ in paths:
        if not os.path.isdir(p_):
            continue
        txt = sh(f"grep -rhoE 'addToRunTimeSelectionTable\\(\\s*{table}\\s*,\\s*[A-Za-z0-9_]+' {p_} 2>/dev/null")
        out |= set(re.findall(r'addToRunTimeSelectionTable\(\s*' + table + r'\s*,\s*([A-Za-z0-9_]+)', txt))
    return out


def schemes(path):
    """make*SurfaceInterpolationScheme(X, ...) -> X: the limiter names a divScheme actually writes."""
    txt = sh(f"grep -rhoE 'make(Limited|LimitedV|Limited01)?SurfaceInterpolationScheme\\([A-Za-z0-9]+' {path} 2>/dev/null")
    return set(re.findall(r'Scheme\(([A-Za-z0-9]+)', txt))


def dirnames(path, drop=()):
    if not os.path.isdir(path):
        return set()
    return {d for d in os.listdir(path)
            if os.path.isdir(os.path.join(path, d)) and d not in drop
            and not d.startswith("derived") and not d.endswith("Model")}


_ALL_BRAE_STRINGS = None


def brae_strings(relpaths=None, pattern=None):
    """Every quoted literal anywhere in brae's source.

    DELIBERATELY BROAD. The manifest's load-bearing claim is the `missing` column -- a type whose
    name appears NOWHERE in brae cannot possibly be handled, so the case must be refused. Scanning
    only the file where a category "should" be parsed produced false missing entries the moment a
    type was handled somewhere else (the div-scheme names are split across the scheme parser and the
    device kernels). A name appearing here means "recognised somewhere", which is the weaker and
    honest claim; whether it is handled CORRECTLY is what the phase gates test, not this script."""
    global _ALL_BRAE_STRINGS
    if _ALL_BRAE_STRINGS is None:
        txt = sh(f"grep -rhoE '\"[A-Za-z][A-Za-z0-9_]{{2,}}\"' {BRAE}/src 2>/dev/null")
        _ALL_BRAE_STRINGS = set(re.findall(r'"([A-Za-z][A-Za-z0-9_]{2,})"', txt))
    return _ALL_BRAE_STRINGS


def demanded(tutroot, category):
    """Type names the tutorials write FOR THIS CATEGORY, scoped to the dictionaries that select it.

    Scoping matters more than it sounds. A scan over every file in every tutorial reported `Gamma` (a
    limiter) as demanded because it appears in a finite-area faSchemes; `midPoint` because a sampling
    dict names it; `coded` and `viscousDissipation` because they are function objects, not fvOptions;
    and `fan` as a boundary condition because it is a PATCH NAME whose type is fixedFluxPressure. Four
    of the six "actionable" entries were noise, which is worse than no list at all -- a manifest that
    cries wolf gets ignored exactly when it is right.

    So each category reads only where OpenFOAM would look for it."""
    if not os.path.isdir(tutroot):
        return set()
    # (file glob, regex capturing the selected name)
    scopes = {
        "fvPatchField":   ([r"0/*", r"0.orig/*"],           r"\btype\s+([A-Za-z][A-Za-z0-9_]*)\s*;"),
        "ddtScheme":      ([r"system/fvSchemes"],           r"ddtSchemes[^}]*?default\s+([A-Za-z][A-Za-z0-9_]*)"),
        "limitedScheme":  ([r"system/fvSchemes"],           r"Gauss\s+([A-Za-z][A-Za-z0-9_]*)"),
        "RASModel":       ([r"constant/turbulenceProperties*", r"constant/momentumTransport*"],
                                                            r"RASModel\s+([A-Za-z][A-Za-z0-9_]*)\s*;"),
        "LESModel":       ([r"constant/turbulenceProperties*", r"constant/momentumTransport*"],
                                                            r"LESModel\s+([A-Za-z][A-Za-z0-9_]*)\s*;"),
        "LESdelta":       ([r"constant/turbulenceProperties*", r"constant/momentumTransport*"],
                                                            r"\bdelta\s+([A-Za-z][A-Za-z0-9_]*)\s*;"),
        "fvOption":       ([r"constant/fvOptions", r"system/fvOptions"],
                                                            r"\btype\s+([A-Za-z][A-Za-z0-9_]*)\s*;"),
        "dynamicFvMesh":  ([r"constant/dynamicMeshDict"],    r"dynamicFvMesh\s+([A-Za-z][A-Za-z0-9_]*)\s*;"),
        "motionSolver":   ([r"constant/dynamicMeshDict"],    r"(?:motionSolver|solver)\s+([A-Za-z][A-Za-z0-9_]*)\s*;"),
    }
    globs, rx = scopes.get(category, ([], None))
    if not rx:
        return set()
    out = set()
    for gl in globs:
        # every tutorial case, at any nesting depth, that has this dictionary
        txt = sh(f"find {tutroot} -path '*/{gl}' -type f -exec cat {{}} + 2>/dev/null")
        out |= set(re.findall(rx, txt, re.S))
    return out


def build(ofsrc):
    fv = f"{ofsrc}/finiteVolume"
    tm = f"{ofsrc}/TurbulenceModels"
    cats = {}

    cats["fvPatchField"] = (
        typenames([f"{fv}/fields/fvPatchFields/basic",
                   f"{fv}/fields/fvPatchFields/constraint",
                   f"{fv}/fields/fvPatchFields/derived",
                   f"{tm}/turbulenceModels/derivedFvPatchFields",
                   f"{tm}/incompressible"]),
        brae_strings(),
    )
    cats["ddtScheme"] = (
        typenames([f"{fv}/finiteVolume/ddtSchemes"]),
        brae_strings(),
    )
    cats["limitedScheme"] = (
        schemes(f"{fv}/interpolation/surfaceInterpolation"),
        brae_strings(),
    )
    cats["RASModel"] = (
        dirnames(f"{tm}/turbulenceModels/RAS", drop=("RASModel",)),
        brae_strings(),
    )
    cats["LESModel"] = (
        dirnames(f"{tm}/turbulenceModels/LES", drop=("LESModel", "LESdeltas", "LESfilters")),
        brae_strings(),
    )
    cats["LESdelta"] = (
        dirnames(f"{tm}/turbulenceModels/LES/LESdeltas"),
        brae_strings(),
    )
    cats["fvOption"] = (
        typenames([f"{ofsrc}/fvOptions"]),
        brae_strings(),
    )
    cats["dynamicFvMesh"] = (
        typenames([f"{ofsrc}/dynamicFvMesh"]),
        brae_strings(),
    )
    cats["motionSolver"] = (
        rts([f"{ofsrc}/fvMotionSolver", f"{ofsrc}/dynamicMesh/motionSolvers"], "motionSolver"),
        brae_strings(),
    )
    return cats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--of", default=os.environ.get("FOAM_SRC", "/usr/lib/openfoam/openfoam2412/src"))
    ap.add_argument("--out", default=os.path.join(BRAE, "docs/pimplefoam-coverage.md"))
    a = ap.parse_args()
    if not os.path.isdir(a.of):
        sys.exit(f"OpenFOAM src not found: {a.of}")

    tutroot = os.path.join(os.path.dirname(a.of.rstrip('/')), "tutorials/incompressible/pimpleFoam")
    cats = build(a.of)
    lines = [
        "# pimpleFoam coverage manifest",
        "",
        "GENERATED by `tools/pimplefoam_coverage.py` -- do not hand-edit the tables.",
        "",
        "Every row is a type OpenFOAM v2412 registers with a `TypeName(\"...\")` and that incompressible",
        "pimpleFoam can therefore select from a case dictionary. `missing` means brae does not name it",
        "anywhere: a case using it must be REFUSED at startup, never run to a plausible wrong answer.",
        "",
        "| category | OF types | named | refused | divergent | missing | **demanded** |",
        "|---|---|---|---|---|---|---|",
    ]
    detail = []
    for name, (of, br) in cats.items():
        refused = set(REFUSED.get(name, {}))
        divergent = set(DIVERGENT.get(name, {}))
        named = (of & br) | refused | divergent
        missingSet = of - named
        missing = sorted(missingSet)
        want = missingSet & demanded(tutroot, name)
        lines.append(f"| `{name}` | {len(of)} | {len(of & br)} | {len(refused & of)} | "
                     f"{len(divergent & of)} | {len(missing)} | **{len(want)}** |")
        detail.append((name, missing, sorted(refused & of), sorted(divergent & of), sorted(want)))

    lines += ["", "## Missing, by category", ""]
    for name, missing, refused, divergent, want in detail:
        lines.append(f"### `{name}` -- {len(missing)} missing")
        lines.append("")
        if refused:
            lines.append("Refused on purpose: " + ", ".join(f"`{x}`" for x in refused))
            lines.append("")
        if divergent:
            lines.append("Deliberately divergent: " + ", ".join(f"`{x}`" for x in divergent))
            lines.append("")
        if want:
            lines.append("**Demanded by a tutorial** (the actionable work list): "
                         + ", ".join(f"`{w}`" for w in want))
            lines.append("")
        lines.append("All missing: " + (", ".join(f"`{m}`" for m in missing) if missing else "_none_"))
        lines.append("")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:14]))
    print(f"\nwritten: {a.out}")


if __name__ == "__main__":
    main()
