#!/bin/bash
# Gate 1: brae's enthalpy equation vs OpenFOAM scalarTransportFoam on the same case.
#
# scalarTransportFoam solves div(phi,T) - laplacian(DT,T) = 0. brae solves
# div(phi,he) - laplacian(alphaEff,he) = 0, and with hConst he = Cp*T + Hf is linear, so the two are the
# same equation when alphaEff == DT. That is why the case sets alpha = DT and alphat = 0.
set -e
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
SRC=${SRC:-$(cd "$(dirname "$0")/energyBox" && pwd)}
BUILD=${BUILD:-$(cd "$(dirname "$0")/../build" && pwd)}
WORK=${WORK:-/tmp/brae_energy_vs_of}

if [ ! -f "$OFBASHRC" ]; then echo "OpenFOAM not found at $OFBASHRC -- skipping"; exit 77; fi
source "$OFBASHRC" 2>/dev/null || true
rm -rf "$WORK"; mkdir -p "$WORK"
cp -r "$SRC"/* "$WORK/"
mkdir -p "$WORK/0" && cp "$WORK"/0.orig/* "$WORK/0/"
cd "$WORK"
blockMesh > log.blockMesh 2>&1
scalarTransportFoam > log.scalarTransportFoam 2>&1
LAST=$(ls -d [0-9]* | grep -vx 0 | sort -g | tail -1)
"$BUILD/test_energy_frozen" "$WORK" "$WORK/$LAST"
