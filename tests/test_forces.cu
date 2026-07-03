// Forces gate, cf wallForces (pressure + viscous) vs OpenFOAM v2412 functionObjects::forces on the converged
// pitzDaily282 case (walls upperWall+lowerWall, rhoInf=1). Computes on the OF-converged fields so the comparison
// isolates the force formula (devTwoSymm gradUBoundary, nutkWallFunction wall nut, Sf integration). The same
// formula validated machine-precision on the kOmegaSST case too (cf force == OF to ~1e-7).
// Run: test_forces <caseDir> <time>
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "forces.cuh"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace brae;

int main(int argc, char** argv) {
    const std::string caseDir = argc > 1 ? argv[1] : "validation/pitzDaily282";
    const std::string t       = argc > 2 ? argv[2] : "282";

    PrimitiveMesh m; m.read(caseDir + "/constant/polyMesh");
    FvGeometry g; g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);

    GeometricField<vector> U = buildField<vector>(readField<vector>(caseDir + "/" + t + "/U"), fvp, m.nCells()); U.evaluateBoundary();
    GeometricField<scalar> p = buildField<scalar>(readField<scalar>(caseDir + "/" + t + "/p"), fvp, m.nCells()); p.evaluateBoundary();
    const std::vector<scalar> kInt = readField<scalar>(caseDir + "/" + t + "/k").internalField;

    const scalar nu = 1e-5;                                  // pitzDaily
    const ForceResult F = wallForces(U, p, kInt, nu, m, g, fvp, {"upperWall", "lowerWall"},
                                     /*rhoRef*/1.0, /*pRef*/0.0, vector{0,0,0}, /*Cmu=betaStar*/0.09);
    // OF moment.dat (about CofR=0), z component: total -5.6981040981117016e-06
    const scalar ofMz = -5.6981040981117016e-06;
    const scalar cfMz = F.moment().z;
    std::printf("  moment_z  cf=%.6e  OF=%.6e  rel %.3e\n", cfMz, ofMz, std::fabs(cfMz-ofMz)/std::fabs(ofMz));

    // OF reference (functionObjects::forces on pitzDaily282/282, walls upperWall+lowerWall, rhoInf=1)
    const vector ofP{3.3474650810742524e-04, 7.5407012675565558e-05, 0.0};
    const vector ofV{9.6854392484701835e-05, -1.6026798002180800e-06, 0.0};
    const vector ofT = ofP + ofV;

    auto rel = [](const vector& a, const vector& b){ return std::sqrt(magSqr(a-b)) / std::fmax(std::sqrt(magSqr(b)), 1e-300); };
    std::printf("test_forces (%s/%s, walls upperWall+lowerWall, rhoInf=1):\n", caseDir.c_str(), t.c_str());
    std::printf("  pressure  cf=(%.4e %.4e)  OF=(%.4e %.4e)  rel %.3e\n", F.pressure.x, F.pressure.y, ofP.x, ofP.y, rel(F.pressure, ofP));
    std::printf("  viscous   cf=(%.4e %.4e)  OF=(%.4e %.4e)  rel %.3e\n", F.viscous.x,  F.viscous.y,  ofV.x, ofV.y, rel(F.viscous, ofV));
    std::printf("  total     cf=(%.4e %.4e)  OF=(%.4e %.4e)  rel %.3e\n", F.total().x,  F.total().y,  ofT.x, ofT.y, rel(F.total(), ofT));

    int fails = 0;
    auto gate=[&](const char* nm, scalar r, scalar tol){ bool ok=r<=tol; if(!ok)++fails; std::printf("  %-9s rel %.3e (tol %.1e) %s\n", nm, r, tol, ok?"OK":"FAIL"); };
    gate("pressure", rel(F.pressure, ofP), 5e-3);
    gate("viscous",  rel(F.viscous,  ofV), 2e-2);   // wall nut (nutkWallFunction) + gradUBoundary -> looser
    gate("total",    rel(F.total(),  ofT), 5e-3);
    gate("moment_z", std::fabs(cfMz-ofMz)/std::fabs(ofMz), 2e-2);   // Md x f integral vs OF moment.dat
    std::printf("%s\n", fails==0?"PASS":"FAIL");
    return fails==0?0:1;
}
