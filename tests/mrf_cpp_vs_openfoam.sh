#!/usr/bin/env bash
# MRF on the MIRRORED _cpp path, end to end, against real OpenFOAM. No CUDA anywhere in this gate.
#
# mixerVessel2D is a rotating-zone case: cellZone `rotor`, omega 104.72 rad/s about z, kEpsilon,
# `bounded Gauss limitedLinearV 1` momentum and `bounded Gauss limitedLinear 1` turbulence, every patch a
# wall or empty. simpleFoam reaches MRF in three places and this exercises all of them:
# correctBoundaryVelocity (UEqn.H:3), DDt (UEqn.H:8) and makeRelative (pEqn.H:5).
#
# THE CONTROL IS THE POINT. With the zones dropped and everything else identical the case has NO driving
# force at all -- the flow is quiescent and every field goes to zero, so the comparison against OpenFOAM
# reads exactly 1.000. MRF is not a correction here, it is 100% of the physics, and a gate that passes
# with it removed would be measuring nothing. Each hook was checked the same way while porting: removing
# DDt alone costs 158x on the momentum residual and removing makeRelative 30x on pressure.
#
# THE BOUND IS LOOSE AND SAYS SO. brae reaches U 4.8e-02 / p 1.2e-02 / k 5.1e-02 / epsilon 1.7e-01
# against OpenFOAM's converged fields -- an order of magnitude looser than what the kEpsilon and
# kOmegaSST gates hold. Assembled at OpenFOAM's OWN converged state the initial residuals are 10x (U) and
# 52x (p) its own, so there is a real discretisation difference still unaccounted for. It is NOT the
# turbulence closure (which cannot affect the first iteration), NOT any single MRF hook (each is
# load-bearing and directionally right), and NOT the internalFaces classification (this mesh cannot
# discriminate it -- see below). The established CUDA solver is 20x on the same measurement, so the
# rebuilt path is already twice as close; the bound is set where the rebuild actually is, and tightening
# it is the open work.
#
# WHAT THIS CASE CANNOT TEST: OpenFOAM's internalFaces are the faces with EITHER cell in the zone, not
# both. Here the rotor zone is bounded by a PATCH rather than by internal faces, so both readings select
# the same 3024 faces and measure identically. A case whose MRF zone meets the rest of the mesh across
# internal faces is needed to gate that, and this one must not be read as having done so.
set -u
SRC="${1:?case dir}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${MRF_CPP_BIN:-$ROOT/build/test_simple_mrf_cpp}"
[ -x "$BIN" ] || { echo "SKIP: no test_simple_mrf_cpp at $BIN"; exit 77; }
SRC="$(cd "$SRC" && pwd)"
[ -d "$SRC/500" ] || { echo "SKIP: no OpenFOAM converged state at $SRC/500"; exit 77; }

echo "== MRF on: must match OpenFOAM =="
MRF_CPP_TOL=2e-01 "$BIN" "$SRC" 0 500 500 || { echo "FAIL: the _cpp MRF did not match OpenFOAM end to end"; exit 1; }

echo "== control: MRF off must NOT match =="
out=$(SIMPLE_MRF_OFF=1 MRF_CPP_TOL=2e-01 "$BIN" "$SRC" 0 500 500 2>&1)
rc=$?
echo "$out" | tail -2
if [ $rc -eq 0 ]; then
    echo "FAIL: the case passes WITHOUT the MRF terms -- this gate measures nothing"
    exit 1
fi
echo "  ok:   the control fails, so the gate is measuring the frame terms"
