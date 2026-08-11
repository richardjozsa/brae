// An `overset` patch must be REFUSED, not treated as a constraint patch.
//
// It used to be listed in isConstraintPatchType, alongside empty/cyclic/wedge/symmetry. Those are patches
// whose fvPatchField OpenFOAM can synthesise from the mesh type alone, which is why a field file may omit
// their boundaryField entry. `overset` is not like that: it is a coupled patch belonging to a solver
// module that replaces the matrix addressing itself (OF's src/overset/fvMeshPrimitiveLduAddressing --
// an acceptor cell's equation becomes an interpolation from donor cells in another mesh region).
//
// Being on that list meant an unsupported overset case RAN: brae synthesised a constraint entry, solved
// the regions as if they were unconnected, converged, and wrote a plausible field set with no message.
// That is the one failure mode brae's contract explicitly rules out, so this is a gate, not a nicety.
//
// The negative control matters as much as the assertion: this must not become "refuse anything unusual".
// A genuine constraint patch (empty) has to keep working, or the fix would break every 2D case.

#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "foam_dict.cuh"
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

using namespace brae;

int main(int argc, char** argv)
{
    const std::string caseDir = argc > 1 ? argv[1] : "validation/pitzDaily";
    int failures = 0;

    // 1. overset must NOT be a constraint patch type.
    if (isConstraintPatchType("overset"))
    {
        std::printf("  FAIL isConstraintPatchType(\"overset\") is true -- a field file omitting its\n"
                    "       boundaryField entry would get a synthesised entry and the case would RUN\n");
        failures++;
    }
    else
    {
        std::printf("  OK   overset is not treated as a constraint patch type\n");
    }

    // 2. NEGATIVE CONTROL: the real constraint types must still be recognised, or every 2D case breaks.
    const char* constraints[] = {"empty", "symmetry", "symmetryPlane", "wedge", "cyclic", "cyclicAMI"};
    bool allOk = true;
    for (const char* c : constraints)
        if (!isConstraintPatchType(c)) { std::printf("  FAIL %s is no longer a constraint type\n", c); allOk = false; }
    if (allOk) std::printf("  OK   empty/symmetry/wedge/cyclic/cyclicAMI still recognised\n");
    else failures++;

    // 3. The real path: a polyMesh whose boundary file declares an overset patch must be refused.
    // Done by rewriting the boundary file rather than mutating the mesh in memory, so this exercises
    // exactly what a user's case would hit.
    {
        const std::string work = "/tmp/brae_overset_refused";
        std::filesystem::remove_all(work);
        std::filesystem::create_directories(work + "/constant");
        std::filesystem::copy(caseDir + "/constant/polyMesh", work + "/constant/polyMesh",
                              std::filesystem::copy_options::recursive);
        // first `type <something>;` inside boundary -> overset
        const std::string bpath = work + "/constant/polyMesh/boundary";
        std::ifstream in(bpath);
        std::string txt((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        in.close();
        const std::size_t t = txt.find("type");
        if (t != std::string::npos)
        {
            const std::size_t semi = txt.find(';', t);
            txt = txt.substr(0, t) + "type            overset" + txt.substr(semi);
            std::ofstream out(bpath);
            out << txt;
        }
        bool refused = false;
        std::string msg;
        try
        {
            PrimitiveMesh m2;
            m2.read(work + "/constant/polyMesh");
            FvGeometry g2;
            g2.build(m2);
            (void)buildPatches(m2, g2);
        }
        catch (const std::exception& e)
        {
            msg = e.what();
            refused = msg.find("overset") != std::string::npos;
        }
        if (refused)
        {
            std::printf("  OK   a polyMesh declaring an overset patch is refused by name\n");
        }
        else
        {
            std::printf("  FAIL an overset patch was accepted -- the case would run and converge WRONG.\n"
                        "       (%s)\n", msg.empty() ? "no exception" : msg.c_str());
            failures++;
        }
    }

    std::printf("overset_refused: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
