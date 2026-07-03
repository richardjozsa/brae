// Phase 1 gate §6.1 (topology), the polyMesh reader reproduces OpenFOAM's mesh counts.
// Reference: v2412 pitzDaily (checkMesh): points 25012, faces 49180, internal 24170,
// cells 12225, 5 patches.  Run: test_mesh_topology <constant/polyMesh>
#include "primitive_mesh.cuh"
#include <cstdio>

int main(int argc, char** argv) {
    if (argc < 2) { std::printf("usage: %s <polyMeshDir>\n", argv[0]); return 2; }

    brae::PrimitiveMesh m;
    m.read(argv[1]);

    int fails = 0;
    auto chk = [&](const char* nm, brae::label got, brae::label exp) {
        const bool ok = (got == exp);
        if (!ok) ++fails;
        std::printf("  %-18s got %8d  expect %8d  %s\n", nm, (int)got, (int)exp, ok ? "OK" : "FAIL");
    };

    std::printf("test_mesh_topology (pitzDaily v2412):\n");
    chk("points",        m.nPoints(),        25012);
    chk("faces",         m.nFaces(),         49180);
    chk("internalFaces", m.nInternalFaces(), 24170);
    chk("cells",         m.nCells(),         12225);
    chk("patches",       (brae::label)m.patches().size(), 5);

    brae::label sumB = 0;
    for (const auto& p : m.patches()) sumB += p.size;
    chk("internal+bndry", m.nInternalFaces() + sumB, m.nFaces());

    // Boundary faces must start exactly at nInternalFaces and be contiguous per patch.
    brae::label expectStart = m.nInternalFaces();
    for (const auto& p : m.patches()) {
        if (p.start != expectStart) {
            std::printf("  FAIL patch %s start %d expect %d\n", p.name.c_str(), (int)p.start, (int)expectStart);
            ++fails;
        }
        expectStart += p.size;
    }

    std::printf("  patches:");
    for (const auto& p : m.patches())
        std::printf(" %s[%s,n=%d@%d]", p.name.c_str(), p.type.c_str(), (int)p.size, (int)p.start);
    std::printf("\n%s\n", fails == 0 ? "PASS" : "FAIL");
    return fails == 0 ? 0 : 1;
}
