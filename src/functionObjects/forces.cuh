#pragma once
// cf force / force-coefficient post-processing, OpenFOAM functionObjects::forces (incompressible), host-side.
// Mirrors forces.C calcForcesMoments (pressure-field branch):
//   fP = rhoRef * Sf * (p_face - pRef)                                  (pressure force per wall face)
//   fV = Sf & devRhoReff,  devRhoReff = -rhoRef * nuEff_face * devTwoSymm(gradU_face)   (viscous)
//   moment = (Cf - CofR) x f
// gradU_face = fvc::gradUBoundary (OF gaussGrad::correctBoundaryConditions: wall-normal = snGrad). nuEff_face =
// nu + nut_wall (nutkWallFunction). p is kinematic; rhoRef = rhoInf (incompressible). Validated vs OF
// `postProcess -func forces` on the converged pitzDaily kOmegaSST case (ctest forces).
#include "cf_types.cuh"
#include "foam_dict.cuh"
#include "geometric_field.cuh"
#include "fvc.cuh"
#include "fv_patch.cuh"
#include "near_wall_dist.cuh"
#include "nut_wall_function.cuh"
#include <algorithm>
#include <cmath>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

namespace brae {

struct ForceResult
{
    vector pressure{0,0,0}, viscous{0,0,0};   // total force = pressure + viscous
    vector momentP{0,0,0},  momentV{0,0,0};   // moment about CofR
    vector total()  const { return pressure + viscous; }
    vector moment() const { return momentP + momentV; }
};

struct ForceCoeffs { scalar Cd = 0, Cl = 0, Cm = 0; };

// The incompressible simpleFoam forceCoeffs contract. Required entries are kept explicit so a malformed
// dictionary cannot accidentally turn into a coefficient with a different normalisation. `rho rhoInf` is
// the only density behaviour available to this incompressible solver; a named volume density would need a
// compressible field and is refused by the caller rather than replaced with rhoInf.
struct ForceCoeffsConfig
{
    std::string name;
    std::vector<std::string> patches;
    std::string executeControl = "timeStep";
    scalar executeInterval = 1;
    std::string writeControl = "timeStep";
    scalar writeInterval = 1;
    std::string rhoName;
    scalar rhoInf = 0, magUInf = 0, Aref = 0, lRef = 0, pRef = 0;
    vector liftDir{0,0,0}, dragDir{0,0,0}, pitchAxis{0,0,0}, CofR{0,0,0};
};

inline std::size_t forceBoundaryFaceCount(const std::vector<FvPatch>& patches)
{
    std::size_t n = 0;
    for (const FvPatch& p : patches)
        if (!isCoupledInterfaceType(p.type)) n += static_cast<std::size_t>(p.size);
    return n;
}

inline vector forceUnitDirection(const vector& v, const std::string& object, const std::string& key)
{
    const scalar m = mag(v);
    if (!(std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z)) || !(std::isfinite(m) && m > 0))
        throw std::runtime_error("forceCoeffs '" + object + "': direction '" + key
                                 + "' must be finite and non-zero");
    return v / m;
}

inline scalar forceRequiredScalar(const FoamDict& d, const std::string& object, const std::string& key)
{
    if (!d.found(key))
        throw std::runtime_error("forceCoeffs '" + object + "': required entry '" + key + "' is missing");
    try { return d.scalarOr(key, 0.0); }
    catch (const std::exception& e)
    {
        throw std::runtime_error("forceCoeffs '" + object + "': entry '" + key + "' is not a scalar (" + e.what() + ")");
    }
}

inline vector forceRequiredVector(const FoamDict& d, const std::string& object, const std::string& key)
{
    if (!d.found(key))
        throw std::runtime_error("forceCoeffs '" + object + "': required entry '" + key + "' is missing");
    const std::vector<scalar> v = d.scalarListOr(key, {});
    if (v.size() != 3)
        throw std::runtime_error("forceCoeffs '" + object + "': entry '" + key + "' must contain exactly three scalars");
    return {v[0], v[1], v[2]};
}

inline ForceCoeffsConfig readForceCoeffsConfig(const std::string& object, const FoamDict& d)
{
    ForceCoeffsConfig c;
    c.name = object;
    if (!d.found("patches"))
        throw std::runtime_error("forceCoeffs '" + object + "': required entry 'patches' is missing");
    c.patches = d.wordListOr("patches", {});
    if (c.patches.empty())
        throw std::runtime_error("forceCoeffs '" + object + "': required entry 'patches' is empty");

    if (!d.found("rho"))
        throw std::runtime_error("forceCoeffs '" + object + "': required entry 'rho' is missing; simpleFoam supports rho rhoInf");
    c.rhoName = d.wordOr("rho", "");
    if (c.rhoName != "rhoInf")
        throw std::runtime_error("forceCoeffs '" + object + "': rho '" + c.rhoName
                                 + "' is unsupported by incompressible simpleFoam; use rho rhoInf");

    c.executeControl  = d.wordOr("executeControl", "timeStep");
    c.executeInterval = d.scalarOr("executeInterval", 1.0);
    c.writeControl    = d.wordOr("writeControl", "timeStep");
    c.writeInterval   = d.scalarOr("writeInterval", 1.0);
    if (c.executeControl != "timeStep" || !std::isfinite(c.executeInterval) || std::fabs(c.executeInterval - 1.0) > 1e-12)
        throw std::runtime_error("forceCoeffs '" + object + "': executeControl/executeInterval '"
                                 + c.executeControl + "/" + std::to_string(c.executeInterval)
                                 + "' is not supported; this implementation samples every SIMPLE iteration"
                                   " and requires executeControl timeStep; executeInterval 1");
    if (c.writeControl != "timeStep" || !std::isfinite(c.writeInterval) || std::fabs(c.writeInterval - 1.0) > 1e-12)
        throw std::runtime_error("forceCoeffs '" + object + "': writeControl/writeInterval '"
                                 + c.writeControl + "/" + std::to_string(c.writeInterval)
                                 + "' is not supported; this implementation writes one history row per SIMPLE"
                                   " iteration and requires writeControl timeStep; writeInterval 1");

    c.rhoInf  = forceRequiredScalar(d, object, "rhoInf");
    c.magUInf = forceRequiredScalar(d, object, "magUInf");
    c.Aref    = forceRequiredScalar(d, object, "Aref");
    c.lRef    = forceRequiredScalar(d, object, "lRef");
    c.liftDir = forceUnitDirection(forceRequiredVector(d, object, "liftDir"), object, "liftDir");
    c.dragDir = forceUnitDirection(forceRequiredVector(d, object, "dragDir"), object, "dragDir");
    c.pitchAxis = forceUnitDirection(forceRequiredVector(d, object, "pitchAxis"), object, "pitchAxis");
    c.CofR      = forceRequiredVector(d, object, "CofR");
    c.pRef = d.scalarOr("pRef", 0.0);

    if (!(std::isfinite(c.rhoInf) && c.rhoInf > 0))
        throw std::runtime_error("forceCoeffs '" + object + "': rhoInf must be finite and > 0");
    if (!(std::isfinite(c.magUInf) && c.magUInf > 0))
        throw std::runtime_error("forceCoeffs '" + object + "': magUInf must be finite and > 0");
    if (!(std::isfinite(c.Aref) && c.Aref > 0))
        throw std::runtime_error("forceCoeffs '" + object + "': Aref must be finite and > 0");
    if (!(std::isfinite(c.lRef) && c.lRef > 0))
        throw std::runtime_error("forceCoeffs '" + object + "': lRef must be finite and > 0");
    if (!(std::isfinite(c.pRef)))
        throw std::runtime_error("forceCoeffs '" + object + "': pRef must be finite");
    if (!(std::isfinite(c.CofR.x) && std::isfinite(c.CofR.y) && std::isfinite(c.CofR.z)))
        throw std::runtime_error("forceCoeffs '" + object + "': CofR must be finite");
    return c;
}

// OpenFOAM's patchSet resolution: exact patch name, then group membership, then regular expression.
// Keep this shared by the host oracle and device selection so a configured patch set cannot diverge between
// the final force calculation and the per-iteration history.
inline bool forcePatchSelected(const FvPatch& wp, const std::vector<std::string>& names)
{
    for (const auto& nm : names)
    {
        if (wp.name == nm) return true;
        for (const auto& gn : wp.inGroups)
            if (gn == nm) return true;
        try
        {
            if (std::regex_match(wp.name, compileFoamRegex(nm))) return true;
        }
        catch (...) {}
    }
    return false;
}

// devTwoSymm(g) = (g + g^T) - (2/3) tr(g) I
inline tensor devTwoSymm(const tensor& g)
{
    tensor s = g + transpose(g);
    const scalar c = (2.0/3.0) * tr(g);
    return {s.xx-c, s.xy, s.xz, s.yx, s.yy-c, s.yz, s.zx, s.zy, s.zz-c};
}

// Pressure + viscous force/moment on the named wall patches. nut_wall via nutkWallFunction(k,y,nu,Cmu,kappa,E)
//, pass the active model's wall Cmu (kEpsilon Cmu, kOmegaSST betaStar). rhoRef = rhoInf, pRef kinematic.
// nutWallBnd (optional): per-boundary-face wall nut to use INSTEAD of nutkWallFunction (e.g. the SA
// nutUSpaldingWallFunction value from the device solver). Indexed in the patch-then-face boundary order.
inline ForceResult wallForces(
    const GeometricField<vector>& U,
    const GeometricField<scalar>& p,
    const std::vector<scalar>& kInternal,
    scalar nu,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches,
    const std::vector<std::string>& wallPatches,
    scalar rhoRef,
    scalar pRef,
    const vector& CofR,
    scalar Cmu = 0.09,
    scalar kappa = 0.41,
    scalar E = 9.8,
    const std::vector<scalar>* nutWallBnd = nullptr)
{
    if (nutWallBnd && nutWallBnd->size() != forceBoundaryFaceCount(patches))
        throw std::runtime_error("forceCoeffs: wall nut has " + std::to_string(nutWallBnd->size())
                                 + " entries, but the non-coupled boundary has "
                                 + std::to_string(forceBoundaryFaceCount(patches)) + " faces");
    const std::vector<tensor> gradC = fvc::gaussGrad(U, m, g, patches);
    const std::vector<std::vector<tensor>> gradB = fvc::gradUBoundary(U, gradC, m, g, patches);
    const std::vector<std::vector<scalar>> yW = nearWallDist(m, g, patches);    // near-wall y for the wall nut
    ForceResult R;
    label bndOff = 0;                                                           // DeviceBoundary order: coupled patches skipped
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const FvPatch& wp = patches[pi];
        if (isCoupledInterfaceType(wp.type)) continue;
        const bool isWall = forcePatchSelected(wp, wallPatches);
        if (!isWall || wp.size == 0)
        {
            bndOff += wp.size;
            continue;
        }
        std::vector<scalar> nutw;
        if (nutWallBnd)
        {
            nutw.resize(wp.size);
            for (label i = 0; i < wp.size; ++i)
                nutw[i] = (*nutWallBnd)[bndOff + i];
        }
        else
            nutw = nutkWallFunction(wp, yW[pi], kInternal, nu, Cmu, kappa, E);
        const std::vector<scalar>& pB = p.boundary[pi]->value();                // p at the wall faces
        for (label i = 0; i < wp.size; ++i)
        {
            const vector Sf = g.Sf()[wp.start + i];
            const vector Cf = g.Cf()[wp.start + i];
            const tensor devReff = (-rhoRef * (nu + nutw[i])) * devTwoSymm(gradB[pi][i]);
            const vector fP = (rhoRef * (pB[i] - pRef)) * Sf;
            const vector fV = dot(Sf, devReff);                                  // Sf & devReff
            const vector Md = Cf - CofR;
            R.pressure += fP;
            R.viscous += fV;
            R.momentP += cross(Md, fP);
            R.momentV += cross(Md, fV);
        }
        bndOff += wp.size;
    }
    return R;
}

// forceCoeffs: Cd/Cl along drag/lift dirs, CmPitch about the pitch axis. q = 0.5*rhoRef*magUInf^2.
// OF forceCoeffs builds a cartesian coordinate system cartesian(origin, e3=liftDir, e1=dragDir) and reports
// CmPitch = moment . e2 with e2 = e3 x e1 = liftDir x dragDir (the DERIVED side axis), NOT the dict `pitchAxis`
// (which OF ignores). For a standard 2D setup liftDir x dragDir = -(0 0 1), so projecting on the dict pitchAxis
// flips the sign. Derive the axis here to match OF. The pitchAxis arg is kept for signature/compat but unused.
inline ForceCoeffs forceCoeffs(
    const ForceResult& F,
    const vector& dragDir,
    const vector& liftDir,
    const vector& pitchAxis,
    scalar rhoRef,
    scalar magUInf,
    scalar Aref,
    scalar lRef)
{
    (void)pitchAxis;
    const vector dDir = forceUnitDirection(dragDir, "<direct>", "dragDir");
    const vector lDir = forceUnitDirection(liftDir, "<direct>", "liftDir");
    const vector pDir = forceUnitDirection(pitchAxis, "<direct>", "pitchAxis");
    const scalar q = 0.5 * rhoRef * magUInf * magUInf;
    const vector sd = cross(lDir, dDir);                             // OF e2 = e3 x e1
    const scalar sdm = mag(sd);
    const vector sideDir = (sdm > 1e-30) ? (sd / sdm) : pDir;       // normalised; fall back to dict if degenerate
    ForceCoeffs c;
    c.Cd = dot(F.total(),  dDir)  / (q * Aref);
    c.Cl = dot(F.total(),  lDir)  / (q * Aref);
    c.Cm = dot(F.moment(), sideDir)  / (q * Aref * lRef);
    return c;
}

} // namespace brae
