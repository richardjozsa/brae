// Cyclic increment C4: rotational transform. On the annular wedge (side1<->side2, rotation about z):
//   (1) GEOMETRY: forwardT applied to (Cf_nbr - centre) coincides with (Cf_own - centre) for every
//       matched face -> the rotation tensor and the face matching are correct (machine precision).
//   (2) VECTOR coupling: a solid-body-rotation field U = omega x r is rotationally symmetric, so the
//       rotated neighbour value must reproduce it -> the coupled face value equals U at the face.
//   (3) SCALAR coupling: a radial scalar is NOT rotated -> coupled value equals the field at the face.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "cyclic_interface.cuh"
#include "cyclic_field.cuh"
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace brae;

int main(int argc, char** argv) {
    const std::string caseDir = argc > 1 ? argv[1] : "validation/cyclicWedge";
    PrimitiveMesh m; m.read(caseDir + "/constant/polyMesh");
    FvGeometry g; g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const std::vector<CyclicInterface> cyclics = buildCyclicInterfaces(m, g, fvp);
    const label nC = m.nCells();

    // (1) geometry: forwardT rotates the neighbour face centre onto the own face centre.
    scalar maxGeo = 0; int nRot = 0;
    for (const CyclicInterface& c : cyclics) {
        if (c.translational) continue; ++nRot;
        const FvPatch& P = fvp[c.patch]; const FvPatch& N = fvp[c.nbrPatch];
        const vector ctr = m.patches()[c.patch].rotationCentre;
        for (label i = 0; i < P.size; ++i) {
            const vector rNbr = dot(g.Cf()[N.start + i] - ctr, transpose(c.forwardT));   // forwardT & (Cf_nbr-ctr)
            maxGeo = std::fmax(maxGeo, mag(rNbr - (g.Cf()[P.start + i] - ctr)));
        }
    }

    // (2) VECTOR transform: a solid-body-rotation field U = omega x r (omega=(0,0,1)) satisfies
    // R*U(P) = U(R*P), so the rotated neighbour value (patchNeighbourField) must equal U at the rotated
    // neighbour cell position. Pure transform, no interpolation -> machine precision.
    std::vector<vector> Uc(nC);
    for (label c = 0; c < nC; ++c) Uc[c] = { -g.C()[c].y, g.C()[c].x, 0.0 };
    GeometricField<vector> U = buildCyclicField<vector>(Uc, fvp, cyclics);
    scalar maxVec = 0;
    for (const CyclicInterface& c : cyclics) {
        if (c.translational) continue;
        const vector ctr = m.patches()[c.patch].rotationCentre;
        const std::vector<vector>& pnf = U.boundary[c.patch]->patchNeighbourField();   // = R * U(nbrCell)
        for (std::size_t i = 0; i < c.nbrFaceCells.size(); ++i) {
            const vector rP = dot(g.C()[c.nbrFaceCells[i]] - ctr, transpose(c.forwardT)) + ctr;  // rotated nbr position
            maxVec = std::fmax(maxVec, mag(pnf[i] - vector{-rP.y, rP.x, 0.0}));               // U at the rotated position
        }
    }

    // (3) SCALAR transform: scalars are NOT rotated -> patchNeighbourField equals the neighbour cell value.
    std::vector<scalar> fc(nC);
    for (label c = 0; c < nC; ++c) fc[c] = g.C()[c].x*g.C()[c].x + g.C()[c].y*g.C()[c].y;
    GeometricField<scalar> F = buildCyclicField<scalar>(fc, fvp, cyclics);
    scalar maxSca = 0;
    for (const CyclicInterface& c : cyclics) {
        if (c.translational) continue;
        const std::vector<scalar>& pnf = F.boundary[c.patch]->patchNeighbourField();
        for (std::size_t i = 0; i < c.nbrFaceCells.size(); ++i) maxSca = std::fmax(maxSca, std::fabs(pnf[i] - fc[c.nbrFaceCells[i]]));
    }

    std::printf("cyclic rotational (nC=%d, %d rotational interfaces):\n", nC, nRot);
    std::printf("  (1) forwardT geometry coincidence : %.3e\n", maxGeo);
    std::printf("  (2) vector transform (solid-body) : %.3e\n", maxVec);
    std::printf("  (3) scalar (unrotated)            : %.3e\n", maxSca);
    const bool pass = nRot == 2 && maxGeo < 1e-12 && maxVec < 1e-12 && maxSca < 1e-12;
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
