#pragma once
// solidBodyMotionFunction -- OF src/dynamicMesh/motionSolvers/displacement/solidBody.
//
// FIRST PIECE of mesh motion, deliberately scoped to what can be verified on its own: the rigid-body
// TRANSFORM. It maps a point at time t and nothing else -- no mesh, no device buffers, no geometry
// recompute. Those are later pieces, and each must be measured against OF the way phi and p0 were.
//
// WHY THIS ORDER. movingWallVelocity (3 pimpleFoam tutorials) needs Uwall = (Cf - oldCf)/deltaT, which
// needs moved points, which needs this. Getting the transform wrong would put a plausible-looking but
// wrong wall velocity into the momentum equation -- exactly the class of silent error that cost this
// project seven retracted findings. So it is built and checked first, alone.
//
// TRANSCRIBED FROM OF, not derived:
//
//   oscillatingLinearMotion.C::transformation()
//       displacement = amplitude*sin(omega*(t + phaseShift)) + verticalShift
//       TR = septernion(-displacement)          i.e. a pure translation
//
//   rotatingMotion.C::transformation()
//       angle = omega->integrate(0, t)          (constant omega -> omega*t)
//       R     = quaternion(axis, angle)
//       TR    = septernion(-origin)*R*septernion(origin)     i.e. rotation ABOUT origin
//
// Only these two are implemented -- they are what the blocked cases declare (oscillatingInletACMI2D,
// rotatingFanInRoom). Every other solidBodyMotionFunction is refused BY NAME rather than approximated
// by one of these: a linear oscillation run where the case asked for a rotation would converge to a
// confident wrong answer.

#include "cf_types.cuh"
#include "foam_dict.cuh"
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <string>

namespace brae {

struct SolidBodyMotion
{
    enum class Kind { None, OscillatingLinear, Rotating };

    Kind   kind        = Kind::None;
    // oscillatingLinearMotion
    vector amplitude{0,0,0};
    scalar omega       = 0;
    scalar phaseShift  = 0;
    vector verticalShift{0,0,0};
    // rotatingMotion
    vector origin{0,0,0};
    vector axis{0,0,1};

    // OF applies the transform to the ORIGINAL (t=0) point positions, not incrementally -- both
    // transformation() functions are absolute functions of t. Accumulating per step would drift.
    vector transform(const vector& p0, scalar t) const
    {
        switch (kind)
        {
            case Kind::OscillatingLinear:
            {
                const scalar s = std::sin(omega*(t + phaseShift));
                return vector{p0.x + amplitude.x*s + verticalShift.x,
                              p0.y + amplitude.y*s + verticalShift.y,
                              p0.z + amplitude.z*s + verticalShift.z};
            }
            case Kind::Rotating:
            {
                // quaternion(axis, angle) applied about `origin`. Rodrigues' rotation is the same
                // rotation the quaternion encodes, written without a quaternion type.
                const scalar angle = omega*t;              // omega->integrate(0,t), constant omega
                const scalar n = std::sqrt(axis.x*axis.x + axis.y*axis.y + axis.z*axis.z);
                if (n <= 0) return p0;
                const vector k{axis.x/n, axis.y/n, axis.z/n};
                const vector r{p0.x - origin.x, p0.y - origin.y, p0.z - origin.z};
                const scalar c = std::cos(angle), s = std::sin(angle);
                const vector kxr{k.y*r.z - k.z*r.y, k.z*r.x - k.x*r.z, k.x*r.y - k.y*r.x};
                const scalar kdr = k.x*r.x + k.y*r.y + k.z*r.z;
                return vector{origin.x + r.x*c + kxr.x*s + k.x*kdr*(1 - c),
                              origin.y + r.y*c + kxr.y*s + k.y*kdr*(1 - c),
                              origin.z + r.z*c + kxr.z*s + k.z*kdr*(1 - c)};
            }
            default: return p0;
        }
    }
};

// What the moving zone is, alongside the motion itself.
struct MeshMotion
{
    bool            active = false;
    std::string     cellZone;    // OF solidBodyMotionSolver moves the points of this zone only
    SolidBodyMotion motion;
};

// constant/dynamicMeshDict -- OF dynamicMotionSolverFvMesh + solidBodyMotionSolver.
//
// Everything outside the implemented scope is refused BY NAME. A case whose mesh does not move the way
// it asked must stop: running dynamicRefineFvMesh as a static mesh, or a rotation as an oscillation,
// converges to a confident wrong answer, which is the failure this codebase refuses on principle.
//
// Absent file -> inactive, which is the correct reading: no dynamicMeshDict means a static mesh.
inline MeshMotion readMeshMotion(const std::string& caseDir)
{
    MeshMotion mm;
    const std::string path = caseDir + "/constant/dynamicMeshDict";
    {
        std::ifstream f(path);
        if (!f.good()) return mm;                      // static mesh: nothing to read
    }
    const FoamDict d = readDict(path);

    const std::string type = d.wordOr("dynamicFvMesh", "");
    if (type != "dynamicMotionSolverFvMesh")
        throw std::runtime_error(
            "brae: constant/dynamicMeshDict dynamicFvMesh '" + type + "' is not implemented (brae "
            "supports dynamicMotionSolverFvMesh with a solidBody motionSolver). Running it as a static "
            "mesh would solve a different problem.");

    const std::string solver = d.wordOr("motionSolver", "");
    if (solver != "solidBody")
        throw std::runtime_error(
            "brae: dynamicMeshDict motionSolver '" + solver + "' is not implemented. Only `solidBody` "
            "(a prescribed rigid transform) is; the others solve a motion equation for the point field, "
            "which brae does not do.");

    // OF motionSolver::coeffDict(): the `<type>Coeffs` sub-dictionary if present, else the dict itself.
    const FoamDict* co = d.subDict("solidBodyCoeffs");
    const FoamDict& c = co ? *co : d;

    mm.cellZone = c.wordOr("cellZone", d.wordOr("cellZone", ""));
    if (mm.cellZone.empty())
        throw std::runtime_error("brae: dynamicMeshDict solidBody has no `cellZone`; brae moves the "
                                 "points of a named zone and will not guess which cells move.");

    const std::string fn = c.wordOr("solidBodyMotionFunction", d.wordOr("solidBodyMotionFunction", ""));
    auto vec3 = [](const std::vector<scalar>& v, vector dflt) {
        return v.size() >= 3 ? vector{v[v.size()-3], v[v.size()-2], v[v.size()-1]} : dflt;
    };
    if (fn == "oscillatingLinearMotion")
    {
        mm.motion.kind          = SolidBodyMotion::Kind::OscillatingLinear;
        mm.motion.amplitude     = vec3(c.scalarListOr("amplitude", d.scalarListOr("amplitude", {})), vector{0,0,0});
        mm.motion.omega         = c.scalarOr("omega", d.scalarOr("omega", 0.0));
        mm.motion.phaseShift    = c.scalarOr("phaseShift", d.scalarOr("phaseShift", 0.0));
        mm.motion.verticalShift = vec3(c.scalarListOr("verticalShift", d.scalarListOr("verticalShift", {})), vector{0,0,0});
    }
    else if (fn == "rotatingMotion")
    {
        mm.motion.kind   = SolidBodyMotion::Kind::Rotating;
        mm.motion.origin = vec3(c.scalarListOr("origin", d.scalarListOr("origin", {})), vector{0,0,0});
        mm.motion.axis   = vec3(c.scalarListOr("axis",   d.scalarListOr("axis",   {})), vector{0,0,1});
        mm.motion.omega  = c.scalarOr("omega", d.scalarOr("omega", 0.0));
    }
    else
        throw std::runtime_error(
            "brae: solidBodyMotionFunction '" + fn + "' is not implemented (brae has "
            "oscillatingLinearMotion and rotatingMotion). Substituting another would move the mesh "
            "differently from the case.");

    mm.active = true;
    return mm;
}

}   // namespace brae
