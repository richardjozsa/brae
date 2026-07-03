#!/usr/bin/env bash
# Self-contained ctest: cf kOmegaSSTLM (transition) vs a SAVED OpenFOAM v2412 oracle (no OF needed at test time).
# Runs cf_gpuSimpleFoam on validation/lmFlatPlate and compares to of_ref/. Pass = U L2 < 3% AND gammaInt mean within
# 5% of OF (the Langtry-Menter transition is captured). Args: $1 = cf_gpuSimpleFoam binary, $2 = lmFlatPlate dir.
set -u
BIN=$1; SRC=$2
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
cp -r "$SRC/0" "$SRC/constant" "$SRC/system" "$W/"
CF_SCALAR_GS=1 CF_SCALAR_LINEARUPWIND=1 "$BIN" "$W" >"$W/log" 2>&1 || { echo "cf run failed"; cat "$W/log"; exit 1; }
grep -qiE "nan" "$W/log" && { echo "FAIL: NaN in cf run"; exit 1; }
CFT=$(ls -d "$W"/[0-9]* 2>/dev/null | grep -v '/0$' | xargs -n1 basename 2>/dev/null | sort -n | tail -1)
[ -z "$CFT" ] && { echo "FAIL: cf wrote no time directory"; exit 1; }
python3 - "$W/$CFT" "$SRC/of_ref" <<'PY'
import sys,re,math,os
def rd(f):
    s=open(f).read(); i=s.find('internalField'); blk=s[i:s.find(';',i)]
    if 'vector' in s[i:i+60]:
        return [tuple(float(x) for x in m.split()) for m in re.findall(r'\(([-\d.eE+ ]+)\)',blk)]
    return [float(x) for x in re.findall(r'[-\d.][-\d.eE+]*',blk[blk.find('('):]) if abs(float(x))<1e30]
cf,of=sys.argv[1],sys.argv[2]
def l2(a,b):
    n=min(len(a),len(b))
    if isinstance(a[0],tuple):
        num=sum(sum((a[i][k]-b[i][k])**2 for k in range(3)) for i in range(n))
        den=sum(sum(b[i][k]**2 for k in range(3)) for i in range(n))
    else:
        num=sum((a[i]-b[i])**2 for i in range(n)); den=sum(b[i]**2 for i in range(n))
    return math.sqrt(num/(den+1e-30))
uL2=l2(rd(f'{cf}/U'),rd(f'{of}/U'))
g_cf=rd(f'{cf}/gammaInt'); g_of=rd(f'{of}/gammaInt')
gm_cf=sum(g_cf)/len(g_cf); gm_of=sum(g_of)/len(g_of)
print(f"  U L2rel = {uL2*100:.2f}%   gammaInt mean cf={gm_cf:.3f} OF={gm_of:.3f}")
ok = uL2 < 0.03 and abs(gm_cf-gm_of) < 0.05
print("  PASS" if ok else "  FAIL (U L2 >= 3% or gammaInt off > 0.05)")
sys.exit(0 if ok else 1)
PY
