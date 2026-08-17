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
    // staticFvMesh is OpenFOAM's explicit "this mesh does not move". A case may name it to be clear
    // rather than to omit dynamicMeshDict, so it is ACCEPTED as no motion -- refusing it would reject a
    // case brae handles perfectly, which is the opposite failure to the one the refusals exist for.
    if (type == "staticFvMesh") return mm;
    if (type != "dynamicMotionSolverFvMesh")
        throw std::runtime_error(
            "brae: constant/dynamicMeshDict dynamicFvMesh '" + type + "' is not implemented (brae has "
            "dynamicMotionSolverFvMesh, and staticFvMesh as a no-op). Running it as a static mesh would "
            "solve a different problem.");

    // OF motionSolver::New reads the name with getCompat<word>("motionSolver", {{"solver", -1666}}) --
    // `solver` is the older key and still what several v2412 tutorials write (RAS/
    // oscillatingInletPeriodicAMI2D among them). Reading only the new one turned an implemented motion
    // into "motionSolver '' is not implemented".
    std::string solver = d.wordOr("motionSolver", "");
    if (solver.empty()) solver = d.wordOr("solver", "");
    // velocityComponentLaplacian is a DIFFERENT motion solver, handled by its own reader
    // (velocity_component_laplacian.cuh). Returning inactive here lets the caller try that one instead
    // of this refusing on its behalf.
    if (solver == "velocityComponentLaplacian") return mm;
    if (solver != "solidBody")
        throw std::runtime_error(
            "brae: dynamicMeshDict motionSolver '" + solver + "' is not implemented. brae has "
            "`solidBody` (a prescribed rigid transform) and `velocityComponentLaplacian` (a Laplace "
            "equation for one component of the point motion). The rest solve a different motion "
            "equation, so the mesh would deform differently -- refused rather than substituted.");

    // OF motionSolver::coeffDict(): the `<type>Coeffs` sub-dictionary if present, else the dict itself.
    const FoamDict* co = d.subDict("solidBodyCoeffs");
    const FoamDict& c = co ? *co : d;

    mm.cellZone = c.wordOr("cellZone", d.wordOr("cellZone", ""));
    if (mm.cellZone.empty())
        throw std::runtime_error("brae: dynamicMeshDict solidBody has no `cellZone`; brae moves the "
                                 "points of a named zone and will not guess which cells move.");

    const std::string fn = c.wordOr("solidBodyMotionFunction", d.wordOr("solidBodyMotionFunction", ""));
    // OF solidBodyMotionFunction::New stores SBMFCoeffs_(dict.optionalSubDict(typeName + "Coeffs")):
    // the FUNCTION's coefficients live in `<function>Coeffs`, nested inside solidBodyCoeffs, not beside
    // `cellZone`. brae read them from solidBodyCoeffs directly, found nothing, and took the defaults --
    // amplitude (0 0 0) and omega 0. The motion function was still RECOGNISED, so nothing refused and
    // nothing warned: the case simply ran with a mesh that never moved.
    //
    // Measured on pimpleFoam/RAS/oscillatingInletPeriodicAMI2D, whose channel should oscillate at
    // amplitude*omega = 0.5*3.14 = 1.57: OpenFOAM's moving walls carried U up to 1.5699 on 223 of 272
    // faces while brae's were identically zero, and 0 of 22356 mesh points moved on any step.
    const FoamDict* fc = fn.empty() ? nullptr : c.subDict(fn + "Coeffs");
    const FoamDict& k = fc ? *fc : c;
    auto vec3 = [](const std::vector<scalar>& v, vector dflt) {
        return v.size() >= 3 ? vector{v[v.size()-3], v[v.size()-2], v[v.size()-1]} : dflt;
    };
    // A recognised motion that evaluates to NO motion is nearly always a coefficient that was not found,
    // which is exactly the failure above. Refuse instead of running a silently static mesh.
    auto checkLive = [&](bool live, const std::string& what)
    {
        if (!live)
            throw std::runtime_error(
                "brae: solidBodyMotionFunction '" + fn + "' has " + what + ", so the mesh would not move "
                "at all. That is almost always a coefficient brae failed to find -- OpenFOAM keeps them "
                "in a `" + fn + "Coeffs` sub-dictionary. Refused rather than run as a static mesh.");
    };
    if (fn == "oscillatingLinearMotion")
    {
        mm.motion.kind          = SolidBodyMotion::Kind::OscillatingLinear;
        mm.motion.amplitude     = vec3(k.scalarListOr("amplitude", c.scalarListOr("amplitude", d.scalarListOr("amplitude", {}))), vector{0,0,0});
        mm.motion.omega         = k.scalarOr("omega", c.scalarOr("omega", d.scalarOr("omega", 0.0)));
        mm.motion.phaseShift    = k.scalarOr("phaseShift", c.scalarOr("phaseShift", d.scalarOr("phaseShift", 0.0)));
        mm.motion.verticalShift = vec3(k.scalarListOr("verticalShift", c.scalarListOr("verticalShift", d.scalarListOr("verticalShift", {}))), vector{0,0,0});
        checkLive(mag(mm.motion.amplitude) > 0 && mm.motion.omega != 0, "zero amplitude or zero omega");
    }
    else if (fn == "rotatingMotion")
    {
        mm.motion.kind   = SolidBodyMotion::Kind::Rotating;
        mm.motion.origin = vec3(k.scalarListOr("origin", d.scalarListOr("origin", {})), vector{0,0,0});
        mm.motion.axis   = vec3(k.scalarListOr("axis",   d.scalarListOr("axis",   {})), vector{0,0,1});
        mm.motion.omega  = k.scalarOr("omega", c.scalarOr("omega", d.scalarOr("omega", 0.0)));
    }
    else
        throw std::runtime_error(
            "brae: solidBodyMotionFunction '" + fn + "' is not implemented (brae has "
            "oscillatingLinearMotion and rotatingMotion). Substituting another would move the mesh "
            "differently from the case.");

    mm.active = true;
    return mm;
}

// Which POINTS move -- OF zoneMotion.C.
//
//     movePts = bitSet(nPoints)
//     for celli in cellZone:  for facei in cells[celli]:  movePts.set(faces[facei])
//     pointIDs_ = movePts.sortedToc()
//     moveAllCells_ = pointIDs_.empty()        <-- an EMPTY selection means move EVERYTHING
//
// Every vertex of every face of every zone cell, so the zone's outer shell of points moves with it and
// the cells just outside DEFORM rather than tear. Selecting only "points of zone cells" would give the
// same set here, but this is what OF walks and the face route is what handles a zone whose boundary
// cuts through a cell.
//
// Returns an empty list to mean "move all points", matching moveAllCells_.
inline std::vector<label> movingPointIDs(const PrimitiveMesh& m, const std::vector<label>& zoneCells)
{
    if (zoneCells.empty()) return {};                  // OF: empty -> move the entire mesh

    // cell -> faces, the inverse of owner/neighbour, which is all brae stores.
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();
    std::vector<std::vector<label>> cellFaces(static_cast<std::size_t>(m.nCells()));
    for (label f = 0; f < m.nFaces(); ++f)
    {
        if (own[f] >= 0 && own[f] < m.nCells()) cellFaces[own[f]].push_back(f);
        if (f < (label)nei.size() && nei[f] >= 0 && nei[f] < m.nCells()) cellFaces[nei[f]].push_back(f);
    }

    std::vector<char> movePts(static_cast<std::size_t>(m.nPoints()), 0);
    for (const label c : zoneCells)
    {
        if (c < 0 || c >= m.nCells()) continue;
        for (const label f : cellFaces[c])
            for (label k = 0; k < m.faceSize(f); ++k)
                movePts[static_cast<std::size_t>(m.faceVert(f, k))] = 1;
    }
    std::vector<label> ids;
    for (label pI = 0; pI < m.nPoints(); ++pI) if (movePts[pI]) ids.push_back(pI);
    return ids;
}

// OF solidBodyMotionSolver::curPoints(): moving points come from points0 transformed ABSOLUTELY;
// every other point keeps its CURRENT position. Starting the non-moving ones from points0 instead
// would silently un-do any earlier motion of theirs.
inline std::vector<vector> curPoints(
    const std::vector<vector>& points0,
    const std::vector<vector>& currentPoints,
    const std::vector<label>& pointIDs,
    const SolidBodyMotion& motion,
    scalar t)
{
    if (pointIDs.empty())                              // moveAllCells
    {
        std::vector<vector> out(points0.size());
        for (std::size_t i = 0; i < points0.size(); ++i) out[i] = motion.transform(points0[i], t);
        return out;
    }
    std::vector<vector> out = currentPoints;
    for (const label pI : pointIDs)
        if (pI >= 0 && pI < (label)out.size())
            out[static_cast<std::size_t>(pI)] = motion.transform(points0[static_cast<std::size_t>(pI)], t);
    return out;
}

}   // namespace brae
