// Cyclic increment C3a: the CyclicFvPatchField (coupled patch field) and cyclic-aware fvc. On the
// periodic channel:
//   (1) coupled face value() == w*own + (1-w)*nbr  (here, for an x-uniform field, == field at the face)
//   (2) patchNeighbourField() == field at the matched neighbour cells
//   (3) fvc::gaussGrad of an x-uniform field has ZERO x-component everywhere incl. cyclic cells
//       (the cyclic coupling reproduces the periodic neighbour exactly, no spurious gradient)
//   (4) fvc::gaussGrad(const) == 0 everywhere
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "cyclic_interface.cuh"
#include "cyclic_field.cuh"
#include "fvc.cuh"
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace brae;

int main(int argc, char** argv) {
    const std::string caseDir = argc > 1 ? argv[1] : "validation/cyclicChannel";
    PrimitiveMesh m; m.read(caseDir + "/constant/polyMesh");
    FvGeometry g; g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const std::vector<CyclicInterface> cyclics = buildCyclicInterfaces(m, g, fvp);
    const label nC = m.nCells();

    // psi = 2.5*y + 1.3  (uniform in the cyclic x-direction).
    std::vector<scalar> psi(nC);
    for (label c = 0; c < nC; ++c) psi[c] = 2.5 * g.C()[c].y + 1.3;
    GeometricField<scalar> P = buildCyclicField<scalar>(psi, fvp, cyclics);

    // (1)+(2): coupled value and neighbour field at the cyclic patches.
    scalar maxVal = 0, maxNbr = 0;
    for (const CyclicInterface& c : cyclics) {
        const FvPatch& fp = fvp[c.patch];
        const std::vector<scalar>& val = P.boundary[c.patch]->value();
        const std::vector<scalar>& pnf = P.boundary[c.patch]->patchNeighbourField();
        for (label i = 0; i < fp.size; ++i) {
            maxVal = std::fmax(maxVal, std::fabs(val[i] - (2.5 * g.Cf()[fp.start + i].y + 1.3)));   // face value
            maxNbr = std::fmax(maxNbr, std::fabs(pnf[i] - psi[c.nbrFaceCells[i]]));                 // neighbour cells
        }
    }

    // (3): x-component of the cyclic-direction gradient is zero for an x-uniform field.
    const std::vector<vector> gradP = fvc::gaussGrad(P, m, g, fvp);
    scalar maxGx = 0; for (label c = 0; c < nC; ++c) maxGx = std::fmax(maxGx, std::fabs(gradP[c].x));

    // (4): gradient of a constant field is zero everywhere.
    std::vector<scalar> cst(nC, 7.0);
    GeometricField<scalar> Cst = buildCyclicField<scalar>(cst, fvp, cyclics);
    const std::vector<vector> gradC = fvc::gaussGrad(Cst, m, g, fvp);
    scalar maxGc = 0; for (label c = 0; c < nC; ++c) maxGc = std::fmax(maxGc, std::fmax(std::fabs(gradC[c].x), std::fmax(std::fabs(gradC[c].y), std::fabs(gradC[c].z))));

    std::printf("cyclic fvc (nC=%d):\n", nC);
    std::printf("  (1) coupled face value err : %.3e\n", maxVal);
    std::printf("  (2) patchNeighbourField err: %.3e\n", maxNbr);
    std::printf("  (3) gaussGrad(x-uniform).x : %.3e\n", maxGx);
    std::printf("  (4) gaussGrad(const)       : %.3e\n", maxGc);
    const bool pass = maxVal < 1e-12 && maxNbr < 1e-14 && maxGx < 1e-12 && maxGc < 1e-12;
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
