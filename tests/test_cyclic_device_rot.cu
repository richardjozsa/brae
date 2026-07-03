// Cyclic increment C-dev-rot: the DEVICE rotational cyclic coupling. deviceCyclicFluxRot must apply the SAME
// rotation as OpenFOAM's patchNeighbourField (forwardT·U[nbr]), validated against a host reference flux, and
// against the physical invariant that a solid-body-rotation field U=omega x r reproduces under the transform.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "cyclic_interface.cuh"
#include "device_cyclic.cuh"
#include "device_buffer.cuh"
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace brae;

static vector rot(const tensor& T, const vector& v) {   // forwardT · v
    return { T.xx*v.x + T.xy*v.y + T.xz*v.z, T.yx*v.x + T.yy*v.y + T.yz*v.z, T.zx*v.x + T.zy*v.y + T.zz*v.z };
}

int main(int argc, char** argv) {
    const std::string caseDir = argc > 1 ? argv[1] : "validation/cyclicWedge";
    PrimitiveMesh m; m.read(caseDir + "/constant/polyMesh");
    FvGeometry g; g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const std::vector<CyclicInterface> cyclics = buildCyclicInterfaces(m, g, fvp);
    const label nC = m.nCells();
    int nRot = 0; for (const auto& c : cyclics) if (!c.translational) ++nRot;

    // a deterministic test vector field (NOT axisymmetric, exercises the full off-diagonal rotation mixing)
    std::vector<scalar> ux(nC), uy(nC), uz(nC);
    for (label c = 0; c < nC; ++c) { ux[c] = std::sin(0.3*c) + 0.2*g.C()[c].x; uy[c] = std::cos(0.21*c) - g.C()[c].y; uz[c] = 0.1*c - std::sin(0.05*c); }

    // ---- device: rotational HbyA flux ----
    DeviceCyclic cyc = buildDeviceCyclic(cyclics, g, fvp);
    DeviceBuffer<scalar> dUx, dUy, dUz; dUx.copyFrom(ux); dUy.copyFrom(uy); dUz.copyFrom(uz);
    deviceCyclicFluxRot(cyc, dUx, dUy, dUz);
    const std::vector<scalar> dphi = cyc.phi.host();

    // ---- host reference: phi[j] = (w*U[own] + (1-w)*forwardT·U[nbr]) · Sf  (same flatten order as buildDeviceCyclic) ----
    std::vector<scalar> href; std::vector<vector> hUown, hUnbr;
    for (const auto& c : cyclics) {
        const FvPatch& P = fvp[c.patch];
        for (label i = 0; i < P.size; ++i) {
            const label o = c.faceCells[i], nb = c.nbrFaceCells[i]; const label gf = P.start + i;
            const vector Uo{ ux[o], uy[o], uz[o] }, Un{ ux[nb], uy[nb], uz[nb] };
            const vector Unr = c.translational ? Un : rot(c.forwardT, Un);
            const scalar w = c.weights[i];
            const vector face = { w*Uo.x + (1-w)*Unr.x, w*Uo.y + (1-w)*Unr.y, w*Uo.z + (1-w)*Unr.z };
            const vector Sf = g.Sf()[gf];
            href.push_back(face.x*Sf.x + face.y*Sf.y + face.z*Sf.z);
        }
    }
    scalar maxFlux = 0; for (std::size_t j = 0; j < href.size(); ++j) maxFlux = std::fmax(maxFlux, std::fabs(dphi[j] - href[j]));

    // ---- physical invariant: solid-body U = omega x r (omega=z) -> forwardT·U[nbr] = U at the rotated nbr position ----
    std::vector<scalar> sx(nC), sy(nC), sz(nC);
    for (label c = 0; c < nC; ++c) { sx[c] = -g.C()[c].y; sy[c] = g.C()[c].x; sz[c] = 0.0; }
    scalar maxSolid = 0;
    for (const auto& c : cyclics) {
        if (c.translational) continue;
        const vector ctr = m.patches()[c.patch].rotationCentre;
        for (std::size_t i = 0; i < c.nbrFaceCells.size(); ++i) {
            const label nb = c.nbrFaceCells[i];
            const vector Un{ sx[nb], sy[nb], sz[nb] };
            const vector got = rot(c.forwardT, Un);                       // device/OF transform of the neighbour
            const vector rP = rot(c.forwardT, g.C()[nb] - ctr);          // rotated neighbour position (relative to axis)
            const vector want{ -(rP.y + ctr.y), (rP.x + ctr.x), 0.0 };   // U=omega x r at the rotated position
            // omega x r with r = rP+ctr ; ctr on axis so ctr contributes 0 to the in-plane swirl about the centre
            const vector wantC{ -rP.y, rP.x, 0.0 };
            maxSolid = std::fmax(maxSolid, mag(got - wantC));
        }
    }

    // (3) flux CONSERVATION across the periodic pair: phi(side1 face i) must = -phi(side2 face i). buildDeviceCyclic
    // stores interface side1 (faces 0..n1-1) then side2 (faces n1..). For a perfect rotational sector, rotation
    // preserves the dot product so the two-sided flux is conservative; a violation -> the continuity asymmetry.
    scalar maxCons = 0;
    if (cyclics.size() == 2) {
        const std::size_t n1 = cyclics[0].faceCells.size();
        for (std::size_t i = 0; i < n1; ++i) maxCons = std::fmax(maxCons, std::fabs(dphi[i] + dphi[n1 + i]));
    }

    std::printf("cyclic device rotational (nC=%ld, %d rotational interfaces):\n", (long)nC, nRot);
    std::printf("  (1) deviceCyclicFluxRot vs host reference : %.3e\n", maxFlux);
    std::printf("  (2) solid-body transform reproduction     : %.3e\n", maxSolid);
    std::printf("  (3) two-sided flux conservation |p1+p2|   : %.3e\n", maxCons);
    const bool ok = nRot >= 1 && maxFlux < 1e-11 && maxSolid < 1e-10 && maxCons < 1e-11;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
