// Cyclic increment C2: the cyclic-coupled Laplacian operator. Three independent machine-precision
// checks on the periodic channel:
//   (a) matrix Amul == explicit face-flux loop (independent assembly path) for a periodic field
//   (b) row-sum zero: Amul(const) == 0 everywhere (cyclic diag+offdiag balance)
//   (c) symmetry: (x, M y) == (y, M x) (the cyclic coupling is symmetric)
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "cyclic_interface.cuh"
#include "cyclic_assembly.cuh"
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
    const label nC = m.nCells(), nIf = m.nInternalFaces();
    const scalar gamma = 0.37;
    const CyclicMatrix M = assembleCyclicLaplacian(m, g, fvp, cyclics, gamma);

    // A periodic test field psi = cos(2*pi*x) + 0.3*y (periodic in x with the unit period).
    std::vector<scalar> psi(nC), x(nC), yv(nC);
    for (label c = 0; c < nC; ++c) {
        const vector C = g.C()[c];
        psi[c] = std::cos(2.0 * M_PI * C.x) + 0.3 * C.y;
        x[c] = std::sin(1.7 * C.x + 0.5) + C.y;       // arbitrary fields for the symmetry check
        yv[c] = std::cos(2.3 * C.y) - 0.4 * C.x;
    }

    // (a) explicit face-flux reference (independent of the matrix diag/upper/lower path).
    const std::vector<label>& own = m.owner(); const std::vector<label>& nei = m.neighbour();
    std::vector<scalar> ref(nC, 0.0);
    for (label f = 0; f < nIf; ++f) {
        const scalar flux = g.deltaCoeffs()[f] * gamma * g.magSf()[f] * (psi[nei[f]] - psi[own[f]]);
        ref[own[f]] += flux; ref[nei[f]] -= flux;
    }
    for (const CyclicInterface& c : cyclics)
        for (std::size_t i = 0; i < c.faceCells.size(); ++i) {
            const scalar coeff = c.deltaCoeffs[i] * gamma * g.magSf()[fvp[c.patch].start + i];
            ref[c.faceCells[i]] += coeff * (psi[c.nbrFaceCells[i]] - psi[c.faceCells[i]]);
        }
    const std::vector<scalar> Apsi = cyclicAmul(M, m, cyclics, psi);
    scalar nA = 0, dA = 0;
    for (label c = 0; c < nC; ++c) { nA = std::fmax(nA, std::fabs(Apsi[c] - ref[c])); dA = std::fmax(dA, std::fabs(ref[c])); }
    const scalar relA = nA / dA;

    // (b) row-sum zero.
    std::vector<scalar> ones(nC, 1.0);
    const std::vector<scalar> Aones = cyclicAmul(M, m, cyclics, ones);
    scalar maxRow = 0; for (label c = 0; c < nC; ++c) maxRow = std::fmax(maxRow, std::fabs(Aones[c]));

    // (c) symmetry via the bilinear form.
    const std::vector<scalar> My = cyclicAmul(M, m, cyclics, yv);
    const std::vector<scalar> Mx = cyclicAmul(M, m, cyclics, x);
    scalar xMy = 0, yMx = 0; for (label c = 0; c < nC; ++c) { xMy += x[c] * My[c]; yMx += yv[c] * Mx[c]; }
    const scalar symErr = std::fabs(xMy - yMx) / std::fmax(std::fabs(xMy), 1e-300);

    std::printf("cyclic laplacian (nC=%d, %zu interfaces):\n", nC, cyclics.size());
    std::printf("  (a) Amul vs explicit flux : rel=%.3e\n", relA);
    std::printf("  (b) row-sum zero          : max=%.3e\n", maxRow);
    std::printf("  (c) symmetry (x,My)=(y,Mx): rel=%.3e\n", symErr);
    const bool pass = relA < 1e-12 && maxRow < 1e-10 && symErr < 1e-12;
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
