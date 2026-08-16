// The mesh-motion contract: which dynamicMeshDict brae accepts, and that everything else is REFUSED.
//
// A motion solver decides how a prescribed boundary movement is spread through the interior. Substitute
// a different one and the mesh deforms differently, so every cell in the domain sits somewhere else --
// which produces a converged, plausible, wrong answer rather than a failure. This is the same class of
// defect as the coupled patches: silent, and only findable by running a case and not believing it.
//
// So the contract is that an unimplemented motion is a STARTUP REFUSAL naming the type, and the legs
// below pin both halves of it -- what is accepted, and what is not. Leg 4 is the one that keeps the
// refusals honest in the other direction: `staticFvMesh` is OpenFOAM's explicit "this mesh does not
// move", and a case that names it must RUN, not be rejected for naming something brae "does not have".
// Refusing valid input is a real failure mode of a refusal policy, not a safe default.
//
// OF reads the solver name with getCompat<word>("motionSolver", {{"solver", -1666}}): `solver` is the
// older key and several v2412 tutorials still write it. Leg 3 pins that, because reading only the new
// key turned an implemented motion into "motionSolver '' is not implemented" once already.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "foam_dict.cuh"
#include "solid_body_motion.cuh"
#include "velocity_component_laplacian.cuh"
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

using namespace brae;

namespace {
int failures = 0;

void check(bool ok, const char* what)
{
    if (!ok) { std::printf("  FAIL: %s\n", what); ++failures; }
    else       std::printf("  ok:   %s\n", what);
}

// readMeshMotion takes a CASE DIRECTORY, which is the real entry point, so the fixture writes a
// throwaway case rather than a bare dictionary -- the path handling (a missing dynamicMeshDict means
// "static mesh") is part of the contract too.
std::string writeCase(const std::string& body)
{
    const std::string dir = "test_motion_contract.tmpcase";
    std::filesystem::create_directories(dir + "/constant");
    std::ofstream f(dir + "/constant/dynamicMeshDict");
    f << "FoamFile { version 2.0; format ascii; class dictionary; object dynamicMeshDict; }\n" << body;
    return dir;
}

// Returns the refusal text, or "" if the dict was accepted by EITHER reader. Both are tried because
// they share the dictionary: solid_body_motion returns inactive for velocityComponentLaplacian so the
// other reader gets its turn, and a refusal must therefore come from whichever one owns the type.
std::string readMotion(const std::string& body)
{
    const std::string dir = writeCase(body);
    std::string err;
    try
    {
        const MeshMotion mm = readMeshMotion(dir);
        (void)mm;
        const VelocityComponentLaplacianMotion vc =
            readVelocityComponentLaplacian(readDict(dir + "/constant/dynamicMeshDict"));
        (void)vc;
    }
    catch (const std::exception& e) { err = e.what(); }
    std::filesystem::remove_all(dir);
    return err;
}
} // namespace

int main()
{
    std::printf("== mesh-motion contract ==\n");

    // ---- Leg 1: the two motions brae implements are accepted ---------------------------------------
    {
        const std::string a = readMotion(
            "dynamicFvMesh   dynamicMotionSolverFvMesh;\n"
            "motionSolver    solidBody;\n"
            "cellZone        rotor;\n"
            "solidBodyMotionFunction  rotatingMotion;\n"
            "rotatingMotionCoeffs { origin (0 0 0); axis (0 0 1); omega 6.2832; }\n");
        check(a.empty(), "solidBody + dynamicMotionSolverFvMesh is accepted");

        const std::string b = readMotion(
            "dynamicFvMesh   dynamicMotionSolverFvMesh;\n"
            "motionSolver    velocityComponentLaplacian;\n"
            "component       x;\n"
            "diffusivity     directional (1 200 0);\n");
        check(b.empty(), "velocityComponentLaplacian is accepted");
    }

    // ---- Leg 2: an unimplemented motion solver is REFUSED, and named --------------------------------
    {
        const std::string e = readMotion(
            "dynamicFvMesh   dynamicMotionSolverFvMesh;\n"
            "motionSolver    sixDoFRigidBodyMotion;\n");
        check(!e.empty(), "an unimplemented motionSolver is refused");
        check(e.find("sixDoFRigidBodyMotion") != std::string::npos,
              "...and the message names it, so the gap is identifiable from the error alone");
    }
    {
        const std::string e = readMotion("dynamicFvMesh   interfaceTrackingFvMesh;\n");
        check(!e.empty() && e.find("interfaceTrackingFvMesh") != std::string::npos,
              "an unimplemented dynamicFvMesh is refused and named");
    }

    // ---- Leg 3: the legacy `solver` key is still read ----------------------------------------------
    {
        const std::string a = readMotion(
            "dynamicFvMesh   dynamicMotionSolverFvMesh;\n"
            "solver          solidBody;\n"
            "cellZone        rotor;\n"
            "solidBodyMotionFunction  rotatingMotion;\n"
            "rotatingMotionCoeffs { origin (0 0 0); axis (0 0 1); omega 6.2832; }\n");
        check(a.empty(), "the older `solver` key selects the motion, as OF's getCompat does");
    }

    // ---- Leg 4: staticFvMesh is ACCEPTED, not refused -----------------------------------------------
    // The failure mode a refusal policy invites is rejecting valid input. staticFvMesh means the mesh
    // does not move, which brae supports completely.
    {
        const std::string e = readMotion("dynamicFvMesh   staticFvMesh;\n");
        check(e.empty(), "staticFvMesh is accepted as `no motion`, not refused as unimplemented");
    }

    // ---- Leg 5: the diffusivity is part of the contract ---------------------------------------------
    // It decides HOW the prescribed motion spreads, so an unimplemented one deforms the mesh differently
    // and must refuse rather than fall back to uniform.
    {
        const std::string e = readMotion(
            "dynamicFvMesh   dynamicMotionSolverFvMesh;\n"
            "motionSolver    velocityComponentLaplacian;\n"
            "component       x;\n"
            "diffusivity     inverseDistance;\n");
        check(!e.empty() && e.find("inverseDistance") != std::string::npos,
              "an unimplemented motion diffusivity is refused and named");
    }
    {
        const std::string e = readMotion(
            "dynamicFvMesh   dynamicMotionSolverFvMesh;\n"
            "motionSolver    velocityComponentLaplacian;\n"
            "diffusivity     uniform;\n");
        check(!e.empty() && e.find("component") != std::string::npos,
              "...and a missing `component` too: without an axis there is nothing to solve for");
    }

    std::printf(failures ? "== FAILED (%d) ==\n" : "== PASSED ==\n", failures);
    return failures ? 1 : 0;
}
