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
python3 - "$WORK/converged.log" "$CONVERGED" <<'PY'
import re
import sys
from math import sqrt

log = open(sys.argv[1]).read()
case_dir = sys.argv[2]
assert "residualControl U valid components=Ux,Uy" in log, log[:2000]
valid = re.search(r"residualControl U valid components=([^\n]+)", log).group(1).split(",")
assert valid == ["Ux", "Uy"], valid

def mesh_list_body(path):
    text = open(path).read()
    match = re.search(r"(?m)^\s*\d+\s*\n\s*\(", text)
    assert match, path
    return text[match.end():]

def read_points(path):
    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    return [tuple(float(value) for value in match)
            for match in re.findall(rf"\(\s*({number})\s+({number})\s+({number})\s*\)",
                                    mesh_list_body(path))]

def read_faces(path):
    faces = []
    for match in re.finditer(r"(\d+)\s*\(([^()]*)\)", mesh_list_body(path)):
        indices = [int(value) for value in re.findall(r"[-+]?\d+", match.group(2))]
        assert len(indices) == int(match.group(1)), (path, match.group(0))
        faces.append(indices)
    return faces

def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])

def subtract(a, b):
    return tuple(a[i] - b[i] for i in range(3))

def face_area_vector(face, points):
    vertices = [points[index] for index in face]
    if len(vertices) == 3:
        c = cross(subtract(vertices[1], vertices[0]), subtract(vertices[2], vertices[0]))
        return tuple(0.5 * value for value in c)
    centre = tuple(sum(vertex[i] for vertex in vertices) / len(vertices) for i in range(3))
    summed = [0.0, 0.0, 0.0]
    for index, this_point in enumerate(vertices):
        next_point = vertices[(index + 1) % len(vertices)]
        triangle_normal = cross(subtract(next_point, this_point), subtract(centre, this_point))
        for component in range(3):
            summed[component] += triangle_normal[component]
    return tuple(0.5 * value for value in summed)

def mesh_valid_components(poly_mesh):
    boundary = open(poly_mesh + "/boundary").read()
    list_start = re.search(r"(?m)^\s*\d+\s*\n\s*\(", boundary)
    assert list_start, poly_mesh + "/boundary"
    patches = []
    for match in re.finditer(r"(?m)^\s*([A-Za-z0-9_.+-]+)\s*\{\s*(.*?)^\s*\}",
                             boundary[list_start.end():], re.S):
        body = match.group(2)
        type_match = re.search(r"(?m)^\s*type\s+(\S+)\s*;", body)
        faces_match = re.search(r"(?m)^\s*nFaces\s+(\d+)\s*;", body)
        start_match = re.search(r"(?m)^\s*startFace\s+(\d+)\s*;", body)
        assert type_match and faces_match and start_match, match.group(1)
        patches.append((match.group(1), type_match.group(1),
                        int(faces_match.group(1)), int(start_match.group(1))))

    points = read_points(poly_mesh + "/points")
    faces = read_faces(poly_mesh + "/faces")
    valid = [True, True, True]
    empty_axes = []
    for name, patch_type, count, start in patches:
        if patch_type != "empty" or count == 0:
            continue
        assert start + count <= len(faces), (name, start, count, len(faces))
        axis = [0.0, 0.0, 0.0]
        for face in faces[start:start + count]:
            area = face_area_vector(face, points)
            magnitude = sqrt(sum(value * value for value in area))
            assert magnitude > 0.0, (name, face)
            for component in range(3):
                axis[component] += abs(area[component] / magnitude)
        normal = 0 if axis[0] >= axis[1] and axis[0] >= axis[2] else (1 if axis[1] >= axis[2] else 2)
        valid[normal] = False
        empty_axes.append((name, normal, axis))
    return [component for component, is_valid in enumerate(valid)
            if is_valid], empty_axes

expected_indices, empty_axes = mesh_valid_components(case_dir + "/constant/polyMesh")
expected = [["Ux", "Uy", "Uz"][index] for index in expected_indices]
assert valid == expected, (valid, expected, empty_axes)

fv_solution = open(case_dir + "/system/fvSolution").read()
rc_match = re.search(r"residualControl\s*\{([^}]*)\}", fv_solution, re.S)
assert rc_match, fv_solution
rc = rc_match.group(1)

def target(field):
    match = re.search(rf"^\s*{re.escape(field)}\s+([0-9.eE+-]+);", rc, re.M)
    assert match, (field, rc)
    return float(match.group(1))

u_target = target("U")
p_target = target("p")
turb_match = re.search(r'^\s*"\(k\|epsilon\|omega\|f\|v2\)"\s+([0-9.eE+-]+);', rc, re.M)
assert turb_match, rc
turb_target = float(turb_match.group(1))

match = re.search(r"SIMPLE solution converged in (\d+) iterations", log)
assert match, log[-2000:]
iteration = int(match.group(1))
assert iteration < 400, iteration

blocks = re.findall(r"Time = (\d+)\n(.*?)(?=\nTime = \d+\n|\nSIMPLE solution converged)", log, re.S)
last = next((body for number, body in blocks if int(number) == iteration), None)
assert last is not None, iteration

def initial(field):
    field_match = re.search(rf"Solving for {field}, Initial residual = ([^,]+),", last)
    assert field_match, (field, last)
    return float(field_match.group(1))

ux, uy, uz = (initial(field) for field in ("Ux", "Uy", "Uz"))
p = initial("p")
controlled_u = max(ux, uy)
assert ux < u_target and uy < u_target, (ux, uy, u_target)
assert controlled_u < u_target, (controlled_u, u_target)
assert p < p_target, (p, p_target)
transport = {}
for field in ("epsilon", "k"):
    transport[field] = initial(field)
    assert transport[field] < turb_target, (field, transport[field], turb_target)

print(f"converged at iteration {iteration} (<400): OK")
print(f"valid U components={','.join(valid)}: OK")
print(f"U initial residuals Ux={ux:.6g}, Uy={uy:.6g}, Uz={uz:.6g}; "
      f"controlled U=max(Ux,Uy)={controlled_u:.6g} < configured U threshold {u_target:.6g}: OK")
print(f"p initial residual {p:.6g} < configured p threshold {p_target:.6g}: OK")
print("epsilon/k initial residuals at convergence "
      + ", ".join(f"{field}={value:.6g}" for field, value in transport.items())
      + f" < configured turbulence threshold {turb_target:.6g}: OK")
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
