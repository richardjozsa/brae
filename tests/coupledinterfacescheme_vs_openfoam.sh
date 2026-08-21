#!/usr/bin/env bash
# The DIV SCHEME's face weight at a COUPLED interface (cyclic / cyclicAMI), against real OpenFOAM.
#
# WHAT WAS WRONG. OpenFOAM assembles a coupled patch with the interpolation weights of the scheme the
# case NAMED -- gaussConvectionScheme::fvmDiv writes internalCoeffs = phi*w and boundaryCoeffs =
# -phi*(1-w) on every coupled patch, with w from limitedSurfaceInterpolationScheme::weights, and
# LimitedScheme::calcLimiter has a coupled() branch that limits an interface face exactly as it limits an
# internal one (only an UNCOUPLED patch is given the constant limiter 1.0). brae hardcoded UPWIND there --
# ifCoeff = -lap + min(phi,0), diag += lap + max(phi,0), which is the special case w = pos0(phi) -- for
# every case and every field, momentum and all six turbulence scalars alike.
#
# pipeCyclic asks for `bounded Gauss limitedLinearV 1` on U and `bounded Gauss limitedLinear 1` on k and
# epsilon, so brae was solving a different equation from the one the case describes at 500 of its faces.
#
# WHY IT SURVIVED SO LONG. brae's device AMI and its _cpp reference agreed to 1e-16 through sixteen stage
# gates -- because both implemented the same wrong scheme. Only OpenFOAM's own fvMatrix could tell them
# apart, and it did: L2 6.86e-01 on the interface off-diagonal, with the gap equal to phi*(1-w) face by
# face. A stage gate against your own reference cannot find a defect the reference shares.
#
# THE FOUR RUNS, and why fewer would not do. Matching OpenFOAM on the limited case is necessary but not
# sufficient: a case where the two schemes happen to agree would pass it while proving nothing. So the
# same case is run under BOTH schemes by BOTH codes:
#
#     1. brae limited  vs  OpenFOAM limited     must AGREE   (the fix)
#     2. brae upwind   vs  OpenFOAM upwind      must AGREE   (upwind was never broken -- and this is what
#                                                             says the weight is scheme-driven, not tuned)
#     3. OpenFOAM limited vs OpenFOAM upwind    must DIFFER  (the case discriminates the two schemes at
#                                                             all -- if it did not, 1 and 2 are vacuous)
#     4. brae limited  vs  brae upwind          must DIFFER  (brae's answer actually FOLLOWS the scheme
#                                                             word; before this work these two runs were
#                                                             identical at the interface by construction)
#
# and the last one is the control the old code fails: it produced 4's "differ" only from the internal
# faces, and the interface contribution to it was identically zero.
#
# THE SYMMETRY CHECK is independent of OpenFOAM entirely. pipeCyclic is a 45-degree sector of a round pipe
# closed by a ROTATIONAL cyclicAMI, so the net transverse pressure force on the walls must vanish by
# symmetry. Assembling the interface upwind breaks that symmetry -- it is the one term that does not
# respect the periodicity -- and the measured force is a direct read of how much: -1.92e-02 before,
# -2.22e-06 after, against a 1.06 axial force.
set -u
SRC="${1:?pipeCyclic case dir}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
OFBIN=/usr/lib/openfoam/openfoam2412/platforms/linuxARM64GccDPInt32Opt
[ -x "$BRAE" ]                 || { echo "SKIP: no brae binary at $BRAE"; exit 77; }
[ -x "$OFBIN/bin/simpleFoam" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
SRC="$(cd "$SRC" && pwd)"
grep -q 'cyclicAMI' "$SRC/constant/polyMesh/boundary" \
    || { echo "FAIL: $SRC has no coupled interface -- this gate would test nothing"; exit 1; }
grep -q 'limitedLinearV' "$SRC/system/fvSchemes" \
    || { echo "FAIL: $SRC does not ask for limitedLinearV, so the limited runs below are not limited"; exit 1; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
# The full OpenFOAM environment, not a hand-assembled subset of it. pipeCyclic's momentum and
# coordinate-transform functionObjects resolve $WM_OPTIONS through OpenFOAM's own etc/controlDict, so the
# PATH+LD_LIBRARY_PATH shortcut the older gates use dies here with "Unknown variable 'WM_OPTIONS'".
# `set +u` around it: OpenFOAM's bashrc reads unset variables by design, so sourcing it under -u aborts
# the script before the first run.
set +u
# shellcheck disable=SC1091
source /usr/lib/openfoam/openfoam2412/etc/bashrc > /dev/null 2>&1 || true
set -u

# One case builder, two schemes. `upwind` rewrites BOTH the momentum and the turbulence div schemes, so
# the two runs differ in the scheme and in nothing else.
mkcase()
{
    cp -r "$SRC" "$1"
    rm -rf "$1"/[1-9]* "$1"/0.[0-9]* "$1"/postProcessing "$1"/log.* "$1"/processor*
    python3 - "$1" "$2" <<'PY'
import re, sys
d, scheme = sys.argv[1], sys.argv[2]
p = d + '/system/fvSchemes'; s = open(p).read()
if scheme == 'upwind':
    s = re.sub(r'div\(phi,U\)\s+[^;]+;',  'div(phi,U)      bounded Gauss upwind;', s)
    s = re.sub(r'turbulence\s+[^;]+;',    'turbulence      bounded Gauss upwind;', s)
open(p, 'w').write(s)
c = d + '/system/controlDict'; s = open(c).read()
s = re.sub(r'writeFormat\s+\S+;',   'writeFormat     ascii;', s)
s = re.sub(r'writePrecision\s+\S+;','writePrecision  15;', s)
s = re.sub(r'functions\s*\{.*?\n\}', 'functions\n{\n}', s, flags=re.S)
open(c, 'w').write(s)
PY
}

for s in limited upwind; do
    mkcase "$W/of_$s" "$s"
    ( cd "$W/of_$s" && timeout 1800 "$OFBIN/bin/simpleFoam" > log.of 2>&1 ) \
        || { echo "FAIL: OpenFOAM did not run the $s case"; tail -5 "$W/of_$s/log.of"; exit 1; }
    mkcase "$W/brae_$s" "$s"
    ( cd "$W/brae_$s" && "$BRAE" . > run.log 2>&1 ) \
        || { echo "FAIL: brae refused or crashed on the $s case"; tail -15 "$W/brae_$s/run.log"; exit 1; }
done

python3 - "$W" <<'PY'
import os, re, sys
import numpy as np
W = sys.argv[1]

def lastTime(d):
    ts = [x for x in os.listdir(d) if re.fullmatch(r'[0-9]+(\.[0-9]+)?', x) and x != '0']
    return None if not ts else max(ts, key=float)

def read(d, f):
    b = open(os.path.join(d, f), 'rb').read()
    m = re.search(rb'internalField\s+nonuniform\s+List<(scalar|vector)>\s*\n(\d+)\s*\n\(', b)
    if not m:
        m2 = re.search(rb'internalField\s+uniform\s+([^;]+);', b)
        return None if not m2 else np.array([[float(x) for x in re.findall(rb'[-+0-9.eE]+', m2.group(1))]])
    typ = m.group(1).decode(); n = int(m.group(2)); start = m.end()
    nc = 3 if typ == 'vector' else 1
    fm = re.search(r'format\s+(\w+)', b[:1024].decode('latin-1'))
    if fm and fm.group(1) == 'binary':
        return np.frombuffer(b[start:start+n*nc*8], dtype='<f8').reshape(n, nc)
    txt = b[start:].decode('latin-1')
    vals = re.findall(r'[-+0-9.eE]+', txt.split(')\n;')[0] if ')\n;' in txt else txt)
    return np.array([float(x) for x in vals[:n*nc]]).reshape(n, nc)

FIELDS = ('U', 'p', 'k', 'epsilon', 'nut')
dirs = {}
for k in ('of_limited', 'of_upwind', 'brae_limited', 'brae_upwind'):
    t = lastTime(os.path.join(W, k))
    if t is None:
        print("  FAIL: %s wrote no result" % k); sys.exit(1)
    dirs[k] = os.path.join(W, k, t)

def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))

rc = 0

# 1 and 2: brae must match OpenFOAM under EACH scheme. The limited bounds are the loose ones -- these are
# two independently converged steady states of a turbulent case, not one step from a shared start -- but
# they are far tighter than the upwind-interface run could reach: it sat at U 5.9e-02, p 2.4e-01,
# k 2.3e-01, so a regression to the old assembly fails the U bound by a factor of 12.
BOUND = {'limited': {'U': 5e-03, 'p': 2e-02, 'k': 8e-02, 'epsilon': 5e-02, 'nut': 8e-02},
         'upwind':  {'U': 5e-03, 'p': 2e-02, 'k': 8e-02, 'epsilon': 5e-02, 'nut': 8e-02}}
for s in ('limited', 'upwind'):
    print("  %d. brae vs OpenFOAM, div(phi,U) = %s" % (1 if s == 'limited' else 2, s))
    for f in FIELDS:
        e = rel(read(dirs['brae_' + s], f), read(dirs['of_' + s], f))
        ok = e < BOUND[s][f]
        print("     %-8s L2 rel %.3e   %s" % (f, e, "ok" if ok else "FAIL (> %.0e)" % BOUND[s][f]))
        if not ok: rc = 1

# 3 and 4: the two schemes must actually give different answers -- in OpenFOAM, so the comparison above
# is not vacuous, and in brae, so brae's answer is shown to follow the scheme word rather than ignore it.
print("  3. OpenFOAM limited vs OpenFOAM upwind (the case must discriminate the schemes)")
for f in ('U', 'k'):
    e = rel(read(dirs['of_limited'], f), read(dirs['of_upwind'], f))
    ok = e > 1e-03
    print("     %-8s L2 rel %.3e   %s" % (f, e, "ok" if ok else "FAIL: the schemes agree, so 1 and 2 prove nothing"))
    if not ok: rc = 1
print("  4. brae limited vs brae upwind (brae's answer must FOLLOW the scheme word)")
for f in ('U', 'k'):
    e = rel(read(dirs['brae_limited'], f), read(dirs['brae_upwind'], f))
    ok = e > 1e-03
    print("     %-8s L2 rel %.3e   %s" % (f, e, "ok" if ok else "FAIL: brae gives the same answer either way"))
    if not ok: rc = 1

sys.exit(rc)
PY
rc=$?

# THE SYMMETRY CHECK. Rotationally periodic sector: the transverse pressure force must vanish. brae prints
# the wall force itself, so this reads brae's own output and never consults OpenFOAM.
FY=$(grep '^forces' "$W/brae_limited/run.log" | tail -1 | sed -n 's/.*pressure=(\([^)]*\)).*/\1/p' | awk '{print $2}')
FZ=$(grep '^forces' "$W/brae_limited/run.log" | tail -1 | sed -n 's/.*pressure=(\([^)]*\)).*/\1/p' | awk '{print $3}')
echo "  5. rotational symmetry: transverse pressure force $FY against axial $FZ"
python3 -c "
import sys
fy, fz = abs(float('$FY')), abs(float('$FZ'))
r = fy/max(fz,1e-300)
ok = r < 1e-04
print('     |Fy|/|Fz| = %.3e   %s' % (r, 'ok' if ok else 'FAIL (>1e-04: the interface breaks the periodicity)'))
sys.exit(0 if ok else 1)
" || rc=1

[ $rc -eq 0 ] && echo "  ok:   brae assembles a coupled interface with the scheme the case names"
exit $rc
