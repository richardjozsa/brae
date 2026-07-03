// Cyclic increment C1: cyclic interface geometry (face matching + coupled deltaCoeffs + weights).
// On the uniform 10x10 periodic channel the coupled deltaCoeffs must be 1/dx = 10 and the symmetric
// interpolation weight 0.5, the period separation must be constant (|sep|=1), and the stored face
// ordering must match geometrically (Cf_nbr[i]-Cf_own[i] == separation for all i).
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "cyclic_interface.cuh"
#include <cmath>
#include <cstdio>
#include <string>

using namespace brae;

int main(int argc, char** argv) {
    const std::string caseDir = argc > 1 ? argv[1] : "validation/cyclicChannel";
    PrimitiveMesh m; m.read(caseDir + "/constant/polyMesh");
    FvGeometry g; g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const std::vector<CyclicInterface> ci = buildCyclicInterfaces(m, g, fvp);

    std::printf("cyclic interfaces found: %zu\n", ci.size());
    scalar maxDC = 0, maxW = 0, maxMatch = 0, maxSepVar = 0;
    bool sizeOk = (ci.size() == 2);
    for (const CyclicInterface& c : ci) {
        const FvPatch& P = fvp[c.patch]; const FvPatch& N = fvp[c.nbrPatch];
        for (label i = 0; i < P.size; ++i) {
            maxDC = std::fmax(maxDC, std::fabs(c.deltaCoeffs[i] - 10.0));
            maxW  = std::fmax(maxW,  std::fabs(c.weights[i] - 0.5));
            const vector sep_i = g.Cf()[N.start + i] - g.Cf()[P.start + i];
            maxMatch  = std::fmax(maxMatch, mag(sep_i - c.separation));      // ordering matches geometry
            maxSepVar = std::fmax(maxSepVar, std::fabs(mag(c.separation) - 1.0));
        }
        std::printf("  patch %s -> nbr %s : %d faces, |sep|=%.4f\n",
                    fvp[c.patch].name.c_str(), fvp[c.nbrPatch].name.c_str(), P.size, mag(c.separation));
    }
    std::printf("max|deltaCoeffs-10|=%.3e  max|w-0.5|=%.3e  max match err=%.3e  |sep|-1=%.3e\n",
                maxDC, maxW, maxMatch, maxSepVar);
    const bool pass = sizeOk && maxDC < 1e-12 && maxW < 1e-12 && maxMatch < 1e-12 && maxSepVar < 1e-12;
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
