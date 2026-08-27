#!/usr/bin/env bash
# Adaptive PIMPLE restart contract: the selected next deltaT and global time index are checkpoint state.
# Euler is intentional here: this test isolates controller-state continuity from backward's expected
# first-restarted-step Euler bootstrap (the second old-time field level is not persisted).
set -eu
BIN="${1:?brae_pimpleFoam binary}"
SRC="${2:?committed pitzDaily case dir}"
WORK="${3:?work dir}"

if [ ! -x "$BIN" ] || [ ! -f "$SRC/constant/polyMesh/points" ]; then
    echo "SKIP: binary or fixture absent"; exit 125
fi

setup_case() {
    rm -rf "$1"; mkdir -p "$1"
    cp -r "$SRC/constant" "$SRC/system" "$SRC/0" "$1/"
    python3 - "$1/system/fvSchemes" <<'PY'
import re, sys
p=sys.argv[1]; s=open(p).read()
open(p,'w').write(re.sub(r'ddtSchemes\s*\{[^}]*\}', 'ddtSchemes\n{\n    default Euler;\n}', s, count=1))
PY
    python3 - "$1/system/fvSolution" <<'PY'
import sys
p=sys.argv[1]; s=open(p).read()
if 'PIMPLE' not in s:
    s += '\nPIMPLE\n{\n nOuterCorrectors 2;\n nCorrectors 2;\n nNonOrthogonalCorrectors 0;\n}\n'
open(p,'w').write(s)
PY
    python3 - "$1/system/controlDict" <<'PY'
import re, sys
p=sys.argv[1]; s=open(p).read()
def setv(k,v):
    global s
    if re.search(r'^\s*'+re.escape(k)+r'\s+',s,re.M):
        s=re.sub(r'^\s*'+re.escape(k)+r'\s+[^;]+;',f'{k} {v};',s,count=1,flags=re.M)
    else: s += f'\n{k} {v};\n'
for k,v in [('application','pimpleFoam'),('startFrom','startTime'),('startTime','0'),
            ('endTime','3e-4'),('deltaT','2e-5'),('adjustTimeStep','yes'),
            ('maxCo','0.2'),('maxDeltaT','4e-5'),('writeControl','timeStep'),
            ('writeInterval','10'),('purgeWrite','0'),('writePrecision','17')]: setv(k,v)
# This is a timestep-state test, so remove any function objects and their host pulls.
s=re.sub(r'\nfunctions\s*\{.*\}\s*$', '\n', s, flags=re.S)
open(p,'w').write(s)
PY
}

rm -rf "$WORK"; mkdir -p "$WORK"
setup_case "$WORK/continuous"
cp -r "$WORK/continuous" "$WORK/restart"

"$BIN" -case "$WORK/continuous" >"$WORK/continuous.log" 2>&1 || {
    echo "FAIL: uninterrupted PIMPLE run failed"; tail -20 "$WORK/continuous.log"; exit 1; }

# Stop a second identical trajectory at an actual computed time; the driver always writes its terminal step.
sed -i 's/^endTime .*/endTime 1.5e-4;/' "$WORK/restart/system/controlDict"
"$BIN" -case "$WORK/restart" >"$WORK/part1.log" 2>&1 || {
    echo "FAIL: restart part 1 failed"; tail -20 "$WORK/part1.log"; exit 1; }
split=$(grep '^Time = ' "$WORK/part1.log" | tail -1 | awk '{print $3}')
state="$WORK/restart/$split/uniform/time"
[ -f "$state" ] || { echo "FAIL: adaptive checkpoint did not write $split/uniform/time"; exit 1; }
saved_dt=$(awk '$1=="deltaT" {gsub(";", "", $2); print $2}' "$state")
dict_dt=$(awk '$1=="deltaT" {gsub(";", "", $2); print $2}' "$WORK/restart/system/controlDict")

sed -i 's/^startFrom .*/startFrom latestTime;/; s/^endTime .*/endTime 3e-4;/' "$WORK/restart/system/controlDict"
"$BIN" -case "$WORK/restart" >"$WORK/part2.log" 2>&1 || {
    echo "FAIL: restart part 2 failed"; tail -20 "$WORK/part2.log"; exit 1; }

python3 - "$WORK" "$split" "$saved_dt" "$dict_dt" <<'PY'
import math, os, re, sys
w, split, saved, initial = sys.argv[1], *map(float,sys.argv[2:])
def times(p): return [float(x) for x in re.findall(r'^Time = (\S+)',open(p).read(),re.M)]
p1=times(w+'/part1.log'); p2=times(w+'/part2.log')
steps=[b-a for a,b in zip([0.0]+p1[:-1],p1)]
if len({round(x,14) for x in steps}) < 3:
    raise SystemExit('FAIL: deltaT did not genuinely vary')
restart_log=open(w+'/part2.log').read()
m=re.search(r'^deltaT = (\S+)',restart_log,re.M)
if not m: raise SystemExit('FAIL: restart log did not report its selected initial deltaT')
resumed=float(m.group(1))
tol=2e-12*max(1.0,abs(saved))
print(f'  dictionary initial deltaT={initial:.17g}')
print(f'  checkpoint next deltaT={saved:.17g}')
print(f'  resumed first deltaT={resumed:.17g}')
if math.isclose(saved,initial,rel_tol=1e-6,abs_tol=1e-15):
    raise SystemExit('FAIL: test is non-discriminating; selected deltaT equals dictionary initial value')
if abs(resumed-saved)>tol:
    raise SystemExit(f'FAIL: resumed deltaT {resumed:.17g} != checkpoint deltaT {saved:.17g}')

def numeric_dirs(case):
    out={}
    for n in os.listdir(case):
        try: out[n]=float(n)
        except ValueError: pass
    return out
def rd(p):
    s=open(p,errors='replace').read(); m=re.search(r'internalField\s+nonuniform\s+List<(?:scalar|vector)>\s*\n\d+\s*\n\(',s)
    if not m: raise SystemExit('FAIL: unreadable field '+p)
    e=s.index('\n)',m.end())
    return [float(x) for x in s[m.end():e].replace('(',' ').replace(')',' ').split()]
common=numeric_dirs(w+'/continuous').keys() & numeric_dirs(w+'/restart').keys()
common=[n for n in common if float(n)>split]
if not common: raise SystemExit('FAIL: no common post-split written time for trajectory comparison')
compare=max(common,key=float)
print(f'  comparing trajectory at common written time {compare}')
for field in ('U','p'):
    a=rd(f'{w}/continuous/{compare}/{field}')
    b=rd(f'{w}/restart/{compare}/{field}')
    rel=max(abs(x-y) for x,y in zip(a,b))/max(max(map(abs,a)),1e-30)
    print(f'  {field}: restart-vs-uninterrupted rel={rel:.3e}')
    # Startup revalidates algebraic turbulence boundary state; measured deterministic floor is a few e-9.
    if rel>1e-8: raise SystemExit(f'FAIL: {field} restart trajectory differs by {rel:.3e}')
PY

echo "PASS pimple_restart_state"
