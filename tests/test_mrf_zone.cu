// MRF increment M1: the MRFZone frame flux + Coriolis source. On the box (zone = all cells, rotation
// about z): (1) makeRelative then makeAbsolute is the identity; (2) the frame flux equals
// (Omega x (Cf-origin)) & Sf; (3) a SOLID-BODY field U = Omega x r has ZERO relative flux (the fluid is
// stationary in the rotating frame), the key physical check; (4) addCoriolis = -V*(Omega x U).
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fv_patch_field.cuh"
#include "fvc.cuh"
#include "mrf_zone.cuh"
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace brae;

int main(int argc, char** argv) {
    const std::string caseDir = argc > 1 ? argv[1] : "validation/mrfBox";
    PrimitiveMesh m; m.read(caseDir + "/constant/polyMesh");
    FvGeometry g; g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells(), nIf = m.nInternalFaces();

    std::vector<label> allCells(nC); for (label c = 0; c < nC; ++c) allCells[c] = c;
    const vector axis{0, 0, 1}, origin{0, 0, 0}; const scalar omega = 2.5;
    const MRFZone z = buildMRFZone(m, allCells, axis, omega, origin);

    // solid-body velocity U = Omega x (C - origin); zeroGradient walls / empty.
    GeometricField<vector> U; U.internal.resize(nC);
    for (label c = 0; c < nC; ++c) U.internal[c] = cross(z.Omega, g.C()[c] - origin);
    for (const FvPatch& p : fvp) {
        if (p.type == "empty") U.boundary.push_back(std::make_unique<EmptyPatchField<vector>>(p));
        else                   U.boundary.push_back(std::make_unique<ZeroGradientPatchField<vector>>(p));
    }
    U.evaluateBoundary();

    // (1) identity: relative then absolute restores phi.
    SurfaceScalarField phi = fvc::flux(U, m, g, fvp);
    const SurfaceScalarField phi0 = phi;
    makeRelative(phi, z, g, fvp); makeAbsolute(phi, z, g, fvp);
    scalar maxId = 0; for (label f = 0; f < nIf; ++f) maxId = std::fmax(maxId, std::fabs(phi.internal[f] - phi0.internal[f]));

    // (2) frame flux formula on a zero field.
    SurfaceScalarField zf; zf.internal.assign(nIf, 0.0); zf.boundary.resize(fvp.size());
    for (std::size_t pi = 0; pi < fvp.size(); ++pi) zf.boundary[pi].assign(fvp[pi].size, 0.0);
    makeRelative(zf, z, g, fvp);
    scalar maxFrame = 0;
    for (label f : z.internalFaces) {
        const scalar expect = -dot(cross(z.Omega, g.Cf()[f] - origin), g.Sf()[f]);
        maxFrame = std::fmax(maxFrame, std::fabs(zf.internal[f] - expect));
    }

    // (3) solid-body relative flux is zero on zone internal faces.
    SurfaceScalarField phiRel = fvc::flux(U, m, g, fvp);
    makeRelative(phiRel, z, g, fvp);
    scalar maxRel = 0; for (label f : z.internalFaces) maxRel = std::fmax(maxRel, std::fabs(phiRel.internal[f]));

    // (4) Coriolis source.
    std::vector<vector> src(nC, vector{0, 0, 0});
    addCoriolis(src, U.internal, g.V(), z);
    scalar maxCor = 0;
    for (label c = 0; c < nC; ++c) maxCor = std::fmax(maxCor, mag(src[c] - (-1.0) * (g.V()[c] * cross(z.Omega, U.internal[c]))));

    std::printf("MRF zone (nC=%d, %zu zone internal faces, |Omega|=%.2f):\n", nC, z.internalFaces.size(), mag(z.Omega));
    std::printf("  (1) relative.absolute identity : %.3e\n", maxId);
    std::printf("  (2) frame flux formula         : %.3e\n", maxFrame);
    std::printf("  (3) solid-body relative flux=0 : %.3e\n", maxRel);
    std::printf("  (4) Coriolis source            : %.3e\n", maxCor);
    const bool pass = maxId < 1e-12 && maxFrame < 1e-14 && maxRel < 1e-12 && maxCor < 1e-14;
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
