// Cyclic increment C-dev: the DEVICE cyclic operator vs the validated host. buildDeviceMesh appends each
// cyclic face as an internal face; deviceAmul over that LDU must equal the host cyclicAmul (cyclic_assembly.cuh)
// for a constant-gamma Laplacian, and the device matrix must be SYMMETRIC. This isolates the device cyclic
// addressing/assembly from the SIMPLE loop, if it fails, the appended-face matrix (not the solver) is the bug.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "cyclic_interface.cuh"
#include "cyclic_assembly.cuh"
#include "device_mesh.cuh"
#include "device_ldu.cuh"
#include "device_cyclic.cuh"
#include "device_buffer.cuh"
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
    const scalar gamma = 0.37;

    // ---- host reference operator (separate-interface assembly, validated) ----
    CyclicMatrix M = assembleCyclicLaplacian(m, g, fvp, cyclics, gamma);

    // two deterministic test vectors (also used for the symmetry check <A x, y> == <x, A y>)
    std::vector<scalar> psi(nC), chi(nC);
    for (label c = 0; c < nC; ++c) { psi[c] = std::sin(0.3*c) + 0.5*std::cos(0.11*c); chi[c] = std::cos(0.17*c) - 0.3*std::sin(0.07*c); }
    const std::vector<scalar> yHost = cyclicAmul(M, m, cyclics, psi);

    // ---- device operator (separate cyclic interface, OF updateInterfaceMatrix) ----
    DeviceMesh dm = buildDeviceMesh(m, g, fvp);                 // LDU = mesh internal faces only (no cyclic)
    DeviceCyclic cyc = buildDeviceCyclic(cyclics, g, fvp);      // periodic interface (both sides)
    DeviceBuffer<scalar> gammaf; gammaf.copyFrom(std::vector<scalar>(dm.nInternalFaces, gamma));
    DeviceBuffer<scalar> lD, lU, lL;
    deviceLaplacianCoeffs(dm, gammaf, lD, lU, lL, /*nonOrth*/false);   // interior Laplacian
    DeviceBuffer<scalar> gammaCell; gammaCell.copyFrom(std::vector<scalar>(nC, gamma));
    deviceCyclicAssembleLaplacian(cyc, gammaCell, lD, /*addToDiag*/true);   // fold cyclic diag + build ifCoeff
    DeviceBuffer<scalar> dpsi; dpsi.copyFrom(psi);
    DeviceBuffer<scalar> dchi; dchi.copyFrom(chi);
    DeviceBuffer<scalar> yDevB(nC), AchiB(nC);
    const DeviceLduView A = deviceLduViewCyclic(dm, lD, lU, lL, cyc.n, cyc.ownCell.data(), cyc.nbrCell.data(), cyc.ifCoeff.data());
    deviceAmul(A, dpsi, yDevB);
    deviceAmul(A, dchi, AchiB);
    const std::vector<scalar> yDev = yDevB.host(), Achi = AchiB.host();

    // (1) device Amul == host cyclicAmul
    scalar maxAbs = 0; label worst = -1;
    for (label c = 0; c < nC; ++c) { scalar a = std::fabs(yDev[c] - yHost[c]); if (a > maxAbs) { maxAbs = a; worst = c; } }

    // (2) device matrix symmetry: <A psi, chi> == <psi, A chi>
    scalar axy = 0, xay = 0;
    for (label c = 0; c < nC; ++c) { axy += yDev[c]*chi[c]; xay += psi[c]*Achi[c]; }
    const scalar symErr = std::fabs(axy - xay) / (std::fabs(axy) + 1e-30);

    std::printf("cyclic device op: nC=%ld nIf0=%ld nIf=%d  |Amul_dev-Amul_host|max=%.3e (cell %ld dev=%.6e host=%.6e)  symErr=%.3e\n",
                (long)nC, (long)m.nInternalFaces(), dm.nInternalFaces, maxAbs,
                (long)worst, worst>=0?yDev[worst]:0.0, worst>=0?yHost[worst]:0.0, symErr);
    const bool ok = (maxAbs < 1e-10) && (symErr < 1e-12);
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
