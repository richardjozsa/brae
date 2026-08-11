// Phase 4.3: the alphat wall conditions load, and the one brae does not implement is refused.
//
// Every OpenFOAM rhoSimpleFoam tutorial with turbulence ships a 0/alphat, so without the
// compressible::alphatWallFunction row brae rejects real compressible cases at load with
// "unsupported BC type". That IS the OF name -- unqualified "alphatWallFunction" is not a valid OF type.
//
// The second half is the point of the test. alphatJayatillekeWallFunction is NOT the same condition --
// it adds a thermal-sublayer P-function and gives a different wall heat flux. Accepting it as the simple
// form would run, converge, and be wrong at every wall, which is the failure mode this project keeps
// meeting (the slip farfield, the rho=1 ddt, the dropped rho in alphat). So it must be refused BY NAME,
// and this test fails if someone later makes it silently accepted.

#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include <cstdio>
#include <string>
#include <vector>

using namespace brae;

namespace {

int failures = 0;

// Builds a one-patch alphat FieldData with the given wall type, and reports whether it loaded.
bool loads(
    const std::vector<FvPatch>& fvp,
    label nC,
    const std::string& wallType)
{
    FieldData<scalar> fd;
    fd.internalUniform = true;
    fd.internalUniformValue = 0.0;
    for (const FvPatch& p : fvp)
    {
        PatchFieldData<scalar> d;
        d.name = p.name;
        d.type = (p.type == "wall") ? wallType : (p.type == "empty" ? "empty" : "calculated");
        d.hasValue = (d.type != "empty");
        d.valueUniform = d.hasValue;
        d.uniformValue = 0.0;
        fd.boundary.push_back(d);
    }
    try
    {
        GeometricField<scalar> at = buildField<scalar>(fd, fvp, nC);
        at.evaluateBoundary();
        return true;
    }
    catch (const std::exception&)
    {
        return false;
    }
}

}   // namespace

int main(int argc, char** argv)
{
    const std::string caseDir = argc > 1 ? argv[1] : "validation/rhoBox";
    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);

    if (!loads(fvp, m.nCells(), "compressible::alphatWallFunction"))
    {
        std::printf("  FAIL compressible::alphatWallFunction was rejected -- real OF compressible cases will not load\n");
        failures++;
    }
    else
    {
        std::printf("  OK   compressible::alphatWallFunction loads\n");
    }

    if (loads(fvp, m.nCells(), "compressible::alphatJayatillekeWallFunction"))
    {
        std::printf("  FAIL compressible::alphatJayatillekeWallFunction was ACCEPTED -- it is a different condition "
                    "(thermal-sublayer P-function) and would give the wrong wall heat flux\n");
        failures++;
    }
    else
    {
        std::printf("  OK   compressible::alphatJayatillekeWallFunction refused by name\n");
    }

    std::printf("alphat_bc: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
