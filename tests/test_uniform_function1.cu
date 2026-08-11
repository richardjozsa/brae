// uniformFixedValue with a NON-CONSTANT uniformValue must be refused, not silently degraded.
//
// OF's uniformFixedValue takes a Function1: `constant`, `table`, `polynomial`, `expression`, `coded`, ...
// brae evaluates only the constant forms. The reader used to skip the others and rely on "dispatch throws
// when there is no value" -- which does not hold, because a case that OVERRIDES an earlier entry leaves a
// stale `value` behind:
//
//     coldWall { type fixedValue; value uniform 350; type uniformFixedValue;
//                uniformValue { type expression; expression "..."; } }
//
// hasValue is still true, so the expression silently became the constant 350. squareBendLiq's T walls are
// exactly this shape. A wrong wall temperature converges perfectly happily.
//
// Parse level on purpose: the claim is about what the READER records and what construction does with it,
// and that is deterministic -- no solve, no tolerance, no GPU.

#include "foam_field_reader.cuh"
#include "fv_patch_field.cuh"
#include "fv_patch.cuh"
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

using namespace brae;

namespace {

int failures = 0;

std::string writeField(const std::string& dir, const std::string& patchBody)
{
    std::filesystem::create_directories(dir);
    const std::string path = dir + "/T";
    std::ofstream f(path);
    f << "FoamFile { version 2.0; format ascii; class volScalarField; object T; }\n"
      << "dimensions [0 0 0 1 0 0 0];\n"
      << "internalField uniform 300;\n"
      << "boundaryField\n{\n" << patchBody << "\n}\n";
    return path;
}

}   // namespace

int main()
{
    const std::string base = "/tmp/brae_uniform_function1";
    std::filesystem::remove_all(base);

    // 1. constant form -> parsed, no refusal marker.
    {
        const FieldData<scalar> fd = readField<scalar>(
            writeField(base + "/const", "    wall { type uniformFixedValue; uniformValue constant 300; }"));
        const PatchFieldData<scalar>& b = fd.boundary.at(0);
        if (b.unsupportedFunction1.empty() && b.hasValue && b.uniformValue == 300.0)
            std::printf("  OK   constant uniformValue parses to 300 and is accepted\n");
        else
        { std::printf("  FAIL constant uniformValue mis-parsed (marker='%s' value=%g)\n",
                      b.unsupportedFunction1.c_str(), (double)b.uniformValue); failures++; }
    }

    // 2. expression form WITH a stale value -> must be marked, and must name the Function1.
    {
        const FieldData<scalar> fd = readField<scalar>(writeField(base + "/expr",
            "    wall { type fixedValue; value uniform 350; type uniformFixedValue;\n"
            "           uniformValue { type expression; expression \"300 + 50\"; } }"));
        const PatchFieldData<scalar>& b = fd.boundary.at(0);
        if (b.unsupportedFunction1 == "expression")
        {
            std::printf("  OK   expression uniformValue is marked (named '%s'), not silently dropped\n",
                        b.unsupportedFunction1.c_str());
        }
        else
        {
            std::printf("  FAIL expression uniformValue left marker '%s'; the stale value %g would be used\n",
                        b.unsupportedFunction1.c_str(), (double)b.uniformValue);
            failures++;
        }

        // ...and construction must refuse it. This is the half that actually protects the user: the
        // marker is useless if makePatchField ignores it.
        FvPatch p;
        p.name = "wall";
        p.type = "wall";
        p.size = 1;
        p.faceCells.assign(1, 0);
        p.deltaCoeffs.assign(1, 1.0);
        p.nf.assign(1, vector{1, 0, 0});
        p.magSf.assign(1, 1.0);
        bool refused = false;
        try { (void)makePatchField<scalar>(p, b); }
        catch (const std::exception& e)
        { refused = std::string(e.what()).find("uniformFixedValue") != std::string::npos; }
        if (refused) std::printf("  OK   construction refuses it by name\n");
        else { std::printf("  FAIL construction accepted a non-constant uniformValue\n"); failures++; }
    }

    // 3. NEGATIVE CONTROL: a plain fixedValue must be untouched, or this would refuse ordinary cases.
    {
        const FieldData<scalar> fd = readField<scalar>(
            writeField(base + "/plain", "    wall { type fixedValue; value uniform 321; }"));
        const PatchFieldData<scalar>& b = fd.boundary.at(0);
        if (b.unsupportedFunction1.empty() && b.uniformValue == 321.0)
            std::printf("  OK   plain fixedValue unaffected\n");
        else { std::printf("  FAIL plain fixedValue disturbed\n"); failures++; }
    }

    // 4. pressureInletOutletVelocity with `tangentialVelocity` must be refused, and without it accepted.
    // OF drives the tangential component (refValue = tv - n*(n & tv),
    // pressureInletOutletVelocityFvPatchVectorField.C:135); brae leaves it at zero, so honouring the case
    // would mean solving a swirl-free inlet where the case asked for swirl. The header said "not
    // supported"; nothing enforced it.
    {
        std::filesystem::create_directories(base + "/tv");
        auto writeU = [&](const std::string& dir, const std::string& body)
        {
            std::filesystem::create_directories(dir);
            const std::string path = dir + "/U";
            std::ofstream f(path);
            f << "FoamFile { version 2.0; format ascii; class volVectorField; object U; }\n"
              << "dimensions [0 1 -1 0 0 0 0];\n"
              << "internalField uniform (0 0 0);\n"
              << "boundaryField\n{\n" << body << "\n}\n";
            return path;
        };
        FvPatch p;
        p.name = "outlet";
        p.type = "patch";
        p.size = 1;
        p.faceCells.assign(1, 0);
        p.deltaCoeffs.assign(1, 1.0);
        p.nf.assign(1, vector{1, 0, 0});
        p.magSf.assign(1, 1.0);

        const FieldData<vector> with = readField<vector>(writeU(base + "/tv/with",
            "    outlet { type pressureInletOutletVelocity; tangentialVelocity uniform (0 5 0);\n"
            "             value uniform (0 0 0); }"));
        bool refused = false;
        try { (void)makePatchField<vector>(p, with.boundary.at(0)); }
        catch (const std::exception& e)
        { refused = std::string(e.what()).find("tangentialVelocity") != std::string::npos; }
        if (refused) std::printf("  OK   pressureInletOutletVelocity + tangentialVelocity refused by name\n");
        else { std::printf("  FAIL tangentialVelocity accepted -- the tangential component would be zeroed\n"); failures++; }

        // NEGATIVE CONTROL: the plain form is the one every simpleFoam tutorial uses; it must still build.
        const FieldData<vector> without = readField<vector>(writeU(base + "/tv/without",
            "    outlet { type pressureInletOutletVelocity; value uniform (0 0 0); }"));
        bool built = true;
        try { (void)makePatchField<vector>(p, without.boundary.at(0)); }
        catch (const std::exception&) { built = false; }
        if (built) std::printf("  OK   plain pressureInletOutletVelocity still accepted\n");
        else { std::printf("  FAIL the plain form was refused too -- that breaks every case using it\n"); failures++; }
    }

    std::printf("uniform_function1: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
