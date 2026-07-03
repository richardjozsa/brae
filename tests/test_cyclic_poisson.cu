// Cyclic increment C3b: the cyclic coupling inside an iterative solver. Manufactured solution on the
// periodic channel: pick a periodic field p_exact, form the DISCRETE RHS f = (cyclic Laplacian) p_exact,
// then solve the cyclic system for x and check x == p_exact up to the constant nullspace (a periodic
// pure-Neumann Poisson problem is defined up to an additive constant). Machine-precision recovery proves
// the cyclic interface off-diagonal coupling is consistent across Krylov iterations.
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
    const label nC = m.nCells();

    CyclicMatrix M = assembleCyclicLaplacian(m, g, fvp, cyclics, 1.0);

    // Manufactured periodic field (smooth, periodic in x with the unit period).
    std::vector<scalar> pExact(nC);
    for (label c = 0; c < nC; ++c) { const vector C = g.C()[c]; pExact[c] = std::sin(2.0 * M_PI * C.x) * (0.5 + C.y); }

    // Discrete RHS, then negate to the SPD form (Laplacian diagonal is negative -> solve -M x = -f).
    const std::vector<scalar> f = cyclicAmul(M, m, cyclics, pExact);
    for (label i = 0; i < nC; ++i) M.diag[i] = -M.diag[i];
    for (auto& u : M.upper) u = -u;  for (auto& l : M.lower) l = -l;
    for (auto& v : M.ifCoeff) for (auto& e : v) e = -e;
    std::vector<scalar> b(nC); for (label i = 0; i < nC; ++i) b[i] = -f[i];

    std::vector<scalar> x(nC, 0.0);
    const int iters = cyclicPCG(M, m, cyclics, x, b, 1e-12, 5000);

    // Compare up to the additive constant (subtract the mean of each).
    scalar mx = 0, me = 0; for (label c = 0; c < nC; ++c) { mx += x[c]; me += pExact[c]; }
    mx /= nC; me /= nC;
    scalar nerr = 0, dref = 0;
    for (label c = 0; c < nC; ++c) { const scalar e = (x[c]-mx) - (pExact[c]-me); nerr += e*e; dref += (pExact[c]-me)*(pExact[c]-me); }
    const scalar rel = std::sqrt(nerr / dref);

    std::printf("cyclic poisson (nC=%d, %d CG iters): solution L2 (up to const) = %.3e\n", nC, iters, rel);
    const bool pass = rel < 1e-9;
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
