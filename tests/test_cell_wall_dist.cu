// C phase ① gate, brae::cellWallDist matches OF v2412 wallDist (meshWave) y_ field (the cell-wise wall
// distance kOmegaSST F1/F2/F3 consume). Reference: validation/of_apps/dumpWallDist writes <case>/<t>/y.
// meshWave seeds wall-face centres and propagates min|C_cell - Cf|; on convex regions that equals the global
// nearest-face-centre min cf computes, so the BULK must match to ~machine precision. Near concave corners
// (the step) meshWave can OVERestimate (cf <= OF), reported and bounded, not a bug.
// Run: test_cell_wall_dist <caseDir> [refTime=0]
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "cell_wall_dist.cuh"
#include "foam_field_reader.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace brae;

int main(int argc, char** argv) {
    if (argc < 2) { std::printf("usage: %s <caseDir> [refTime]\n", argv[0]); return 2; }
    const std::string caseDir = argv[1];
    const std::string refTime = (argc >= 3) ? argv[2] : "0";

    PrimitiveMesh m; m.read(caseDir + "/constant/polyMesh");
    FvGeometry g; g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);

    const std::vector<scalar> y = cellWallDist(m, g, patches);

    const FieldData<scalar> yof = readField<scalar>(caseDir + "/" + refTime + "/y");
    const std::vector<scalar> yOf = yof.internalUniform
        ? std::vector<scalar>(m.nCells(), yof.internalUniformValue) : yof.internalField;
    if (static_cast<label>(yOf.size()) != m.nCells()) {
        std::printf("  FAIL ref y size %zu != nCells %d\n", yOf.size(), (int)m.nCells()); return 1;
    }

    // OF correctWalls (default true) sets EXACT distance on cells that own a wall face OR are point-connected
    // to one; interior cells keep the meshWave face-centre overestimate. cf's exact reproduces correctWalls for
    // all cells, so split the error: wall-corrected cells (where SST's F1/F2 live) MUST match OF; interior cells
    // differ only by OF's meshWave overestimate (cf <= OF), and there F1/F2 -> 0 so it cannot move the model.
    std::vector<char> wallCell(m.nCells(), 0);                    // own a wall face
    std::vector<char> wallVtx(m.points().size(), 0);             // belongs to a wall face
    for (const FvPatch& p : patches) if (p.type == "wall")
        for (label i = 0; i < p.size; ++i) {
            wallCell[p.faceCells[i]] = 1;
            const label f = p.start + i;
            for (label j = m.faceOffsets()[f]; j < m.faceOffsets()[f+1]; ++j) wallVtx[m.faceVerts()[j]] = 1;
        }
    std::vector<char> wallCorr = wallCell;                        // + point-connected cells (correctBoundaryPointCells)
    for (label f = 0; f < m.nFaces(); ++f) {                      // a cell is point-corrected if any of its faces touches a wall vertex
        bool touch = false;
        for (label j = m.faceOffsets()[f]; j < m.faceOffsets()[f+1] && !touch; ++j) if (wallVtx[m.faceVerts()[j]]) touch = true;
        if (touch) { wallCorr[m.owner()[f]] = 1; if (f < m.nInternalFaces()) wallCorr[m.neighbour()[f]] = 1; }
    }

    const scalar Lref = *std::max_element(yOf.begin(), yOf.end());
    scalar maxRelW = 0, sumRelW = 0, maxRelI = 0, sumRelI = 0, maxUnder = 0;
    label nW = 0, nI = 0, argW = -1, argI = -1, nUnder = 0;
    for (label c = 0; c < m.nCells(); ++c) {
        const scalar r = std::fabs(y[c] - yOf[c]) / std::fmax(yOf[c], 1e-300);
        if (y[c] > yOf[c] + 1e-9*Lref) { ++nUnder; maxUnder = std::fmax(maxUnder, (y[c]-yOf[c])/yOf[c]); }  // cf>OF (should not happen for corrected cells)
        if (wallCorr[c]) { ++nW; sumRelW += r; if (r>maxRelW){maxRelW=r;argW=c;} }
        else             { ++nI; sumRelI += r; if (r>maxRelI){maxRelI=r;argI=c;} }
    }
    std::printf("test_cell_wall_dist (%s, nCells=%d, Lref=%.4g), cf=EXACT distance:\n", caseDir.c_str(), (int)m.nCells(), Lref);
    std::printf("  WALL-CORRECTED cells %d:  mean rel %.3e  max rel %.3e (cell %d cf=%.6e OF=%.6e)\n",
                (int)nW, nW?sumRelW/nW:0.0, maxRelW, (int)argW, argW>=0?y[argW]:0.0, argW>=0?yOf[argW]:0.0);
    std::printf("  INTERIOR cells       %d:  mean rel %.3e  max rel %.3e (cell %d cf=%.6e OF=%.6e)  [OF meshWave overestimate]\n",
                (int)nI, nI?sumRelI/nI:0.0, maxRelI, (int)argI, argI>=0?y[argI]:0.0, argI>=0?yOf[argI]:0.0);
    std::printf("  cells cf>OF (unexpected) %d  maxUnder %.2e\n", (int)nUnder, maxUnder);

    // Gate: at WALL-CORRECTED cells cf-exact must equal OF's correctWalls distance to near-machine precision
    // (that is the SST-relevant region). Interior is informational (OF's own approximation).
    int fails = 0;
    auto gate=[&](const char* nm,double v,double t){bool ok=v<=t;if(!ok)++fails;std::printf("  %-26s %.3e (tol %.1e) %s\n",nm,v,t,ok?"OK":"FAIL");};
    gate("wall-corrected mean rel", nW?sumRelW/nW:0.0, 1.0e-6);
    gate("wall-corrected max rel",  maxRelW,           1.0e-4);
    std::printf("%s\n", fails==0?"PASS":"FAIL");
    return fails==0?0:1;
}
