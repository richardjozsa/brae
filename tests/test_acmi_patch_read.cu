// cyclicACMI: reading nonOverlapPatch off the polyMesh boundary, and keeping it through the mesh cache.
//
// WHAT ACMI NEEDS THIS FOR. A cyclicACMI patch is only half of an interface. The other half is a WALL
// patch, geometrically COINCIDENT with it (OF's blockMeshDict gives the two the same faces -- verified
// on the generated mesh of pimpleFoam/RAS/oscillatingInletACMI2D, where the couple and blockage patches
// agree to |dCf| = 0 and |dArea| = 0). The AMI overlap fraction splits the area between them:
//
//     coupled  Sf = Sf_raw * max(tol, mask)                    (cyclicACMIPolyPatch.C:252)
//     wall     Sf = Sf_raw * (1 - min(max(mask,tol), 1-tol))   (cyclicACMIPolyPatch.C:226)
//
// so a face that has slid off its neighbour becomes solid wall instead of leaking. Which wall patch
// takes that area is named by the `nonOverlapPatch` keyword, and brae's boundary parser used to drop
// it: unknown keys are skipped silently (primitive_mesh.cu, the trailing else). Without it there is no
// way to know which patch to hand the uncovered area to.
//
// WHY THE CACHE LEG IS NOT PADDING. PatchInfo is serialised into the binary mesh cache
// (.brae_meshcache), which is auto-loaded whenever it is newer than owner -- no env var needed. A new
// field that is parsed but NOT serialised produces the worst kind of bug: correct on a cold run, empty
// on every warm one, i.e. an ACMI case that works until someone caches the mesh. Leg 3 writes the
// cache, then EDITS THE ASCII FILE UNDERNEATH IT, so a value that still reads correctly can only have
// come from the binary blob -- otherwise the leg would pass without the cache being used at all.
//
// Leg 4 covers the other half: the record layout CHANGED when this field was added, so a cache written
// by the previous build must be REJECTED (magic CFM2 -> CFM3) rather than replayed as garbage.
#include "primitive_mesh.cuh"
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace brae;

namespace {

int failures = 0;

void hdr(std::ofstream& f, const char* cls, const char* obj)
{
    f << "FoamFile\n{\n    version 2.0;\n    format ascii;\n    class " << cls
      << ";\n    location \"constant/polyMesh\";\n    object " << obj << ";\n}\n";
}

// A single hex cell, six boundary faces, no internal faces. Deliberately minimal: this exercises the
// BOUNDARY PARSER, and nothing here depends on the mesh being physically sensible.
void writeMesh(const std::filesystem::path& d, const char* nonOverlap1)
{
    std::filesystem::create_directories(d);
    {
        std::ofstream f(d/"points");
        hdr(f, "vectorField", "points");
        f << "8\n(\n(0 0 0)\n(1 0 0)\n(1 1 0)\n(0 1 0)\n(0 0 1)\n(1 0 1)\n(1 1 1)\n(0 1 1)\n)\n";
    }
    {
        std::ofstream f(d/"faces");
        hdr(f, "faceList", "faces");
        f << "6\n(\n4(0 3 2 1)\n4(4 5 6 7)\n4(0 1 5 4)\n4(2 3 7 6)\n4(0 4 7 3)\n4(1 2 6 5)\n)\n";
    }
    {
        std::ofstream f(d/"owner");
        hdr(f, "labelList", "owner");
        f << "6\n(\n0\n0\n0\n0\n0\n0\n)\n";
    }
    {
        std::ofstream f(d/"neighbour");
        hdr(f, "labelList", "neighbour");
        f << "0\n(\n)\n";
    }
    {
        std::ofstream f(d/"boundary");
        hdr(f, "polyBoundaryMesh", "boundary");
        f << "5\n(\n"
          << "    ACMI1_couple\n    {\n        type cyclicACMI;\n        inGroups 1(cyclicACMI);\n"
          << "        nFaces 1;\n        startFace 0;\n        neighbourPatch ACMI2_couple;\n"
          << "        AMIMethod faceAreaWeightAMI;\n        requireMatch 0;\n"
          << "        nonOverlapPatch " << nonOverlap1 << ";\n    }\n"
          << "    ACMI2_couple\n    {\n        type cyclicACMI;\n        nFaces 1;\n"
          << "        startFace 1;\n        neighbourPatch ACMI1_couple;\n"
          << "        nonOverlapPatch ACMI2_blockage;\n    }\n"
          << "    ACMI1_blockage\n    {\n        type wall;\n        nFaces 1;\n        startFace 2;\n    }\n"
          << "    ACMI2_blockage\n    {\n        type wall;\n        nFaces 1;\n        startFace 3;\n    }\n"
          << "    outer\n    {\n        type wall;\n        nFaces 2;\n        startFace 4;\n    }\n"
          << ")\n";
    }
}

const PatchInfo* byName(const PrimitiveMesh& m, const std::string& n)
{
    for (const PatchInfo& p : m.patches()) if (p.name == n) return &p;
    return nullptr;
}

void expectNonOverlap(const PrimitiveMesh& m, const char* patch, const char* want, const char* leg)
{
    const PatchInfo* p = byName(m, patch);
    if (!p) { std::printf("  FAIL %s: patch %s missing\n", leg, patch); ++failures; return; }
    if (p->nonOverlapPatch != want)
    {
        std::printf("  FAIL %s: %s nonOverlapPatch = '%s', expected '%s'\n",
                    leg, patch, p->nonOverlapPatch.c_str(), want);
        ++failures;
    }
}

}   // namespace

int main()
{
    const std::filesystem::path root = std::filesystem::path("test_acmi_patch_read.tmp");
    const std::filesystem::path dir  = root/"constant"/"polyMesh";
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    writeMesh(dir, "ACMI1_blockage");

    // ---------------------------------------------------------------------------------------------
    // 1. The keyword is parsed off the boundary file.
    {
        PrimitiveMesh m;
        m.read(dir.string());
        expectNonOverlap(m, "ACMI1_couple", "ACMI1_blockage", "parse");
        expectNonOverlap(m, "ACMI2_couple", "ACMI2_blockage", "parse");

        // ANTI-VACUOUS. If the fixture ever stopped carrying cyclicACMI patches with the keyword, every
        // assertion above would compare "" against "" and pass while testing nothing.
        int acmi = 0;
        for (const PatchInfo& p : m.patches())
            if (p.type == "cyclicACMI" && !p.nonOverlapPatch.empty()) ++acmi;
        if (acmi != 2)
        { std::printf("  FAIL vacuous: fixture has %d cyclicACMI patches carrying the key, expected 2\n", acmi); ++failures; }

        // 2. NO BLEED. Every other patch must be EMPTY -- a parser that leaked the previous patch's
        // value, or filled the field from an unrelated key, would still pass leg 1.
        for (const PatchInfo& p : m.patches())
            if (p.type != "cyclicACMI" && !p.nonOverlapPatch.empty())
            {
                std::printf("  FAIL bleed: non-ACMI patch %s carries nonOverlapPatch '%s'\n",
                            p.name.c_str(), p.nonOverlapPatch.c_str());
                ++failures;
            }
        std::printf("  parse: ACMI1_couple -> '%s', ACMI2_couple -> '%s', %zu patches clean\n",
                    byName(m,"ACMI1_couple")->nonOverlapPatch.c_str(),
                    byName(m,"ACMI2_couple")->nonOverlapPatch.c_str(), m.patches().size());
    }

    // ---------------------------------------------------------------------------------------------
    // 3. The field survives the binary mesh cache -- and is PROVEN to come from it.
    // Write the cache, then rewrite the ASCII boundary with a DIFFERENT value. The cache is still newer
    // than owner, so read() warm-loads it; getting the original value back can then only mean the blob
    // carried the field. Skipping the serialisation would give the edited value instead.
    {
        const std::filesystem::path cache = dir/".brae_meshcache";
        setenv("BRAE_MESH_CACHE", "1", 1);
        { PrimitiveMesh m; m.read(dir.string()); }             // cold parse + write cache
        unsetenv("BRAE_MESH_CACHE");
        if (!std::filesystem::exists(cache))
        { std::printf("  FAIL cache: .brae_meshcache was not written; leg 3 proves nothing\n"); ++failures; }
        else
        {
            writeMesh(dir, "SENTINEL_from_ascii");              // edit ASCII underneath the cache
            std::filesystem::last_write_time(cache, std::filesystem::last_write_time(dir/"owner") + std::chrono::seconds(2));
            PrimitiveMesh m;
            m.read(dir.string());
            const PatchInfo* p = byName(m, "ACMI1_couple");
            if (p && p->nonOverlapPatch == "SENTINEL_from_ascii")
            {
                std::printf("  FAIL cache: read fell back to the ASCII file, so this leg never exercised\n"
                            "       the binary path and says nothing about serialisation\n");
                ++failures;
            }
            else
            {
                expectNonOverlap(m, "ACMI1_couple", "ACMI1_blockage", "cache");
                expectNonOverlap(m, "ACMI2_couple", "ACMI2_blockage", "cache");
                std::printf("  cache: warm-loaded, nonOverlapPatch survived the binary round-trip\n");
            }

            // 4. A cache from a build with the OLD record layout must be REJECTED, not replayed. Flip
            // the magic and require the reader to fall back to the (now edited) ASCII value.
            {
                std::FILE* f = std::fopen(cache.string().c_str(), "r+b");
                if (f) { unsigned bad = 0x43464D32; std::fwrite(&bad, sizeof(bad), 1, f); std::fclose(f); }
                PrimitiveMesh m2;
                m2.read(dir.string());
                const PatchInfo* q = byName(m2, "ACMI1_couple");
                if (!q || q->nonOverlapPatch != "SENTINEL_from_ascii")
                {
                    std::printf("  FAIL stale cache: a CFM2 cache was not rejected (got '%s') -- an old\n"
                                "       cache would be replayed against the new record layout as garbage\n",
                                q ? q->nonOverlapPatch.c_str() : "<missing>");
                    ++failures;
                }
                else
                {
                    std::printf("  stale cache: CFM2 magic rejected, fell back to a cold parse\n");
                }
            }
        }
    }

    std::filesystem::remove_all(root, ec);
    std::printf("test_acmi_patch_read: %d failures\n", failures);
    return failures ? 1 : 0;
}
