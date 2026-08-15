// Writing a time directory when the ORIGINAL field is gzipped.
//
// writeVolField re-reads the input field as a TEMPLATE -- its FoamFile header, dimensions and
// boundaryField shape are echoed into the new time directory -- and it opened that file with a bare
// ifstream. OpenFOAM writes `U.gz` whenever writeCompression is on, and pimpleFoam/LES/periodicPlaneChannel
// ships its 0/ that way, so brae read the mesh and the fields happily (those already go through gzSlurp),
// solved the whole step, and then died at its FIRST write with "cannot read ./0/U".
//
// The fix is one call, but the failure mode is worth a test: everything upstream of the write is
// gzip-aware, so nothing catches this until a run has already done all of its work.
#include "box_mesh.cuh"
#include "foam_field_writer.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using namespace brae;

namespace {
int failures = 0;
}

int main()
{
    const char* tmp = std::getenv("TMPDIR");
    const std::string dir = std::string(tmp ? tmp : "/tmp") + "/brae_gz_write";
    std::system(("rm -rf " + dir + " && mkdir -p " + dir).c_str());

    PrimitiveMesh m = boxtest::boxMesh(2, 2, 1);
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells();

    std::string text =
        "FoamFile { version 2.0; format ascii; class volScalarField; object p; }\n"
        "dimensions      [0 2 -2 0 0 0 0];\n"
        "internalField   uniform 0;\n"
        "boundaryField\n{\n";
    for (const FvPatch& q : fvp) text += "    " + q.name + "\n    {\n        type            zeroGradient;\n    }\n";
    text += "}\n";
    { std::ofstream o(dir + "/p"); o << text; }
    std::system(("gzip -f " + dir + "/p").c_str());   // now only p.gz exists, as OpenFOAM leaves it

    const bool plainGone = !std::ifstream(dir + "/p").good();
    const bool gzThere   =  std::ifstream(dir + "/p.gz").good();
    std::printf("  fixture: plain file gone=%d, p.gz present=%d\n", (int)plainGone, (int)gzThere);
    if (!plainGone || !gzThere)
    { std::printf("  FAIL vacuous: the fixture is not actually gzip-only, so this tests nothing\n"); ++failures; }

    const std::vector<scalar> vals(nC, 1.25);
    bool threw = false;
    std::string msg;
    try { writeVolField<scalar, FvPatch>(dir + "/p", dir + "/p_out", vals, fvp, 12); }
    catch (const std::exception& e) { threw = true; msg = e.what(); }
    if (threw)
    {
        std::printf("  FAIL writeVolField could not use a gzipped template: %s\n", msg.c_str());
        std::printf("       Every read upstream of the write is gzip-aware, so a run reaches this having\n"
                    "       already done all of its solving.\n");
        ++failures;
    }
    else
    {
        std::ifstream out(dir + "/p_out");
        const std::string got((std::istreambuf_iterator<char>(out)), std::istreambuf_iterator<char>());
        std::printf("  wrote %zu bytes\n", got.size());
        if (got.find("1.25") == std::string::npos)
        { std::printf("  FAIL the written field does not carry the values\n"); ++failures; }
        if (got.find("boundaryField") == std::string::npos || got.find("[0 2 -2 0 0 0 0]") == std::string::npos)
        {
            std::printf("  FAIL the template's dimensions/boundaryField did not survive -- the gzipped file\n"
                        "       was opened but not actually parsed\n");
            ++failures;
        }
    }

    std::printf("gz_field_write: %d failures\n", failures);
    return failures ? 1 : 0;
}
