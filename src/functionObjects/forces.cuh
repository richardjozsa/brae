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
#include "geometric_field.cuh"
#include "fvc.cuh"
#include "fv_patch.cuh"
#include "near_wall_dist.cuh"
#include "nut_wall_function.cuh"
#include <string>
#include <vector>

namespace brae {

struct ForceResult {
    vector pressure{0,0,0}, viscous{0,0,0};         // total force = pressure + viscous
    vector momentP{0,0,0},  momentV{0,0,0};         // moment about CofR
    vector total()  const { return pressure + viscous; }
    vector moment() const { return momentP + momentV; }
};

struct ForceCoeffs { scalar Cd = 0, Cl = 0, Cm = 0; };

// devTwoSymm(g) = (g + g^T) - (2/3) tr(g) I
inline tensor devTwoSymm(const tensor& g) {
    tensor s = g + transpose(g);
    const scalar c = (2.0/3.0) * tr(g);
    return {s.xx-c, s.xy, s.xz, s.yx, s.yy-c, s.yz, s.zx, s.zy, s.zz-c};
}

// Pressure + viscous force/moment on the named wall patches. nut_wall via nutkWallFunction(k,y,nu,Cmu,kappa,E)
//, pass the active model's wall Cmu (kEpsilon Cmu, kOmegaSST betaStar). rhoRef = rhoInf, pRef kinematic.
// nutWallBnd (optional): per-boundary-face wall nut to use INSTEAD of nutkWallFunction (e.g. the SA
// nutUSpaldingWallFunction value from the device solver). Indexed in the patch-then-face boundary order.
inline ForceResult wallForces(const GeometricField<vector>& U, const GeometricField<scalar>& p,
                              const std::vector<scalar>& kInternal, scalar nu,
                              const PrimitiveMesh& m, const FvGeometry& g, const std::vector<FvPatch>& patches,
                              const std::vector<std::string>& wallPatches, scalar rhoRef, scalar pRef,
                              const vector& CofR, scalar Cmu = 0.09, scalar kappa = 0.41, scalar E = 9.8,
                              const std::vector<scalar>* nutWallBnd = nullptr) {
    const std::vector<tensor> gradC = fvc::gaussGrad(U, m, g, patches);
    const std::vector<std::vector<tensor>> gradB = fvc::gradUBoundary(U, gradC, m, g, patches);
    const std::vector<std::vector<scalar>> yW = nearWallDist(m, g, patches);    // near-wall y for the wall nut
    ForceResult R;
    label bndOff = 0;                                                           // cumulative boundary-face index (all patches)
    for (std::size_t pi = 0; pi < patches.size(); ++pi) {
        const FvPatch& wp = patches[pi];
        // OF functionObjects::forces resolves its `patches` entry via polyBoundaryMesh::patchSet: each token matches
        // by exact NAME, then by GROUP membership (inGroups), then as a REGEX, so `patches (motorBikeGroup)` or
        // `("motorBike_.*")` selects the bike, not just an exactly-named patch. Reuse the same name->group->regex rule.
        bool isWall = false;
        for (const auto& nm : wallPatches) {
            if (wp.name == nm) { isWall = true; break; }                                         // exact patch name
            bool grp = false; for (const auto& gn : wp.inGroups) if (gn == nm) { grp = true; break; }
            if (grp) { isWall = true; break; }                                                   // group membership
            try { if (std::regex_match(wp.name, compileFoamRegex(nm))) { isWall = true; break; } } catch (...) {}  // regex
        }
        if (!isWall || wp.size == 0) { bndOff += wp.size; continue; }
        std::vector<scalar> nutw;
        if (nutWallBnd) { nutw.resize(wp.size); for (label i = 0; i < wp.size; ++i) nutw[i] = (*nutWallBnd)[bndOff + i]; }
        else nutw = nutkWallFunction(wp, yW[pi], kInternal, nu, Cmu, kappa, E);
        const std::vector<scalar>& pB = p.boundary[pi]->value();                // p at the wall faces
        for (label i = 0; i < wp.size; ++i) {
            const vector Sf = g.Sf()[wp.start + i];
            const vector Cf = g.Cf()[wp.start + i];
            const tensor devReff = (-rhoRef * (nu + nutw[i])) * devTwoSymm(gradB[pi][i]);
            const vector fP = (rhoRef * (pB[i] - pRef)) * Sf;
            const vector fV = dot(Sf, devReff);                                  // Sf & devReff
            const vector Md = Cf - CofR;
            R.pressure += fP;  R.viscous += fV;
            R.momentP  += cross(Md, fP);  R.momentV += cross(Md, fV);
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
inline ForceCoeffs forceCoeffs(const ForceResult& F, const vector& dragDir, const vector& liftDir,
                               const vector& pitchAxis, scalar rhoRef, scalar magUInf, scalar Aref, scalar lRef) {
    (void)pitchAxis;
    const scalar q = 0.5 * rhoRef * magUInf * magUInf;
    const vector sd = cross(liftDir, dragDir);                       // OF e2 = e3 x e1
    const scalar sdm = mag(sd);
    const vector sideDir = (sdm > 1e-30) ? (sd / sdm) : pitchAxis;   // normalised; fall back to dict if degenerate
    ForceCoeffs c;
    c.Cd = dot(F.total(),  dragDir)  / (q * Aref);
    c.Cl = dot(F.total(),  liftDir)  / (q * Aref);
    c.Cm = dot(F.moment(), sideDir)  / (q * Aref * lRef);
    return c;
}

} // namespace brae
