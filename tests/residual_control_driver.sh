#!/usr/bin/env bash
# Driver-level SIMPLE residualControl contract using the validation pitzDaily case.
set -eu

BIN="${1:?brae binary}"
SRC="${2:?pitzDaily fixture}"
WORK="${3:?work directory}"
if [ ! -f "$SRC/constant/polyMesh/points" ] || [ ! -f "$SRC/0/U" ]; then
    echo "SKIP: fixture '$SRC' not present"
    exit 125
fi

rm -rf "$WORK"
mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT

make_case() {
    local d="$1"
    mkdir -p "$d"
    cp -r "$SRC/constant" "$SRC/system" "$SRC/0" "$d/"
    sed -i -E 's/^endTime[[:space:]]+2000;/endTime         400;/' "$d/system/controlDict"
    if ! grep -Eq '^endTime[[:space:]]+400;' "$d/system/controlDict"; then
        echo "FAIL: endTime substitution did not take effect in $d/system/controlDict"
        exit 1
    fi
    # Keep pitzDaily's p and turbulence thresholds, but make U non-gating so the driver
    # test fails if the turbulence convergence loop is removed.
    sed -i -E 's/^        U[[:space:]]+1e-3;/        U               1e-2;/' "$d/system/fvSolution"
    if ! grep -Eq '^        U[[:space:]]+1e-2;' "$d/system/fvSolution"; then
        echo "FAIL: U threshold substitution did not take effect in $d/system/fvSolution"
        exit 1
    fi
}

replace_with_unknown_only() {
    local d="$1"
    python3 - "$d/system/fvSolution" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
start_marker = "    residualControl\n    {"
start = text.find(start_marker)
if start < 0:
    raise SystemExit("FAIL: residualControl block not found")
end = text.find("\n    }", start)
if end < 0:
    raise SystemExit("FAIL: residualControl block end not found")
end += len("\n    }")
replacement = "    residualControl\n    {\n        onlyUnknownField 1e-3;\n    }"
path.write_text(text[:start] + replacement + text[end:])
PY
}

run_case() {
    local d="$1"
    local name="$2"
    "$BIN" -case "$d" > "$WORK/$name.log" 2>&1
}

CONVERGED="$WORK/converged"
make_case "$CONVERGED"
run_case "$CONVERGED" converged
python3 - "$WORK/converged.log" <<'PY'
import re
import sys

log = open(sys.argv[1]).read()
assert "residualControl U valid components=Ux,Uy" in log, log[:2000]
match = re.search(r"SIMPLE solution converged in (\d+) iterations", log)
assert match, log[-2000:]
iteration = int(match.group(1))
assert iteration < 400, iteration

uz_values = re.findall(r"Solving for Uz, Initial residual = ([^,]+),", log)
assert uz_values, log[:2000]
uz_initial = float(uz_values[-1])
assert 0.45 <= uz_initial <= 0.65, uz_initial

blocks = re.findall(r"Time = (\d+)\n(.*?)(?=\nTime = \d+\n|\nSIMPLE solution converged)", log, re.S)
last = next((body for number, body in blocks if int(number) == iteration), None)
assert last is not None, iteration
transport = {}
for field in ("epsilon", "k"):
    field_match = re.search(rf"Solving for {field}, Initial residual = ([^,]+),", last)
    assert field_match, (field, last)
    transport[field] = float(field_match.group(1))
    assert transport[field] <= 1e-3, (field, transport[field])

print(f"converged at iteration {iteration} (<400): OK")
print(f"valid U components Ux,Uy and Uz initial residual {uz_initial:.6g}: OK")
print("epsilon/k initial residuals at convergence "
      + ", ".join(f"{field}={value:.6g}" for field, value in transport.items())
      + ": OK")
PY

UNKNOWN="$WORK/unknown-only"
make_case "$UNKNOWN"
replace_with_unknown_only "$UNKNOWN"
run_case "$UNKNOWN" unknown-only
python3 - "$WORK/unknown-only.log" <<'PY'
import re
import sys

log = open(sys.argv[1]).read()
assert "SIMPLE solution converged" not in log, log[-2000:]
match = re.search(r"SIMPLE reached endTime \((\d+) iterations\)", log)
assert match and int(match.group(1)) == 400, log[-2000:]
print("unknown-only residualControl reaches endTime (400 iterations): OK")
PY
