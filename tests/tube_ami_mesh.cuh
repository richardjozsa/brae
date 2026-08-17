#pragma once
// Shared test fixture: a closed polygonal TUBE joined by a cyclicAMI pair -- the curved-interface
// geometry the rest of the AMI suite lacks (every other AMI fixture here is a flat 8-to-10-face plane).
//
// Two concentric annuli meet at r = R. Each side is built independently, so the two sides of the
// interface can be discretised differently in BOTH directions:
//     nz   -- divisions along the axis      (non-conformity in z)
//     N    -- sectors around the axis       (non-conformity in theta, and DIFFERENT SURFACES)
//
// The N knob is the important one and the two settings mean different things:
//
//   Nin == Nout: both sides are the SAME polygonal tube -- same facets, same corner angles -- so a
//     source face is exactly coplanar with the target faces covering it and the exact coverage is 1.
//     An analytic reference with no tolerance to choose.
//
//   Nin != Nout: the sides are DIFFERENT polygons inscribed in the same cylinder. Their facets are no
//     longer coplanar, there is no closed-form coverage, and the residual is faceting error which must
//     VANISH UNDER REFINEMENT. That is the real-world case (a snappyHexMesh rotor/stator interface) and
//     convergence is what distinguishes faceting from a defect.
//
// Faces are emitted in OpenFOAM order: all internal faces first, in upper-triangular order (owner
// ascending, neighbour ascending within an owner), then one contiguous block per patch. Both blocks'
// internal faces precede every boundary face -- interleaving them leaves neighbour[] misaligned with
// the internal-face range, which shows up as negative cell volumes.
#include "primitive_mesh.cuh"
#include <cmath>
#include <vector>

namespace brae {
namespace tubetest {

// Annuli Rin..R (Nin sectors, nzA in z) and R..Rout (Nout sectors, nzB in z), joined at r = R by the
// cyclicAMI pair AMI_in / AMI_out. Requires N >= 3 (a 2-sector ring has a degenerate wrap face).
inline PrimitiveMesh tubeMesh(
    label Nin, label Nout, label nzA, label nzB,
    scalar Rin, scalar R, scalar Rout, scalar H)
{
    const label nCin = Nin*nzA;
    const label nC   = nCin + Nout*nzB;

    std::vector<vector> pts;
    std::vector<label>  fv, foff, own, nei;
    std::vector<PatchInfo> patches;
    foff.push_back(0);

    // A closed ring has N distinct angles, not N+1: the sector index wraps. The two blocks do NOT
    // share points -- an AMI interface is a mesh discontinuity, exactly as blockMesh duplicates the
    // vertices lying on an AMI plane.
    auto addBlock = [&](label N, label nz, scalar r0, scalar r1) -> label
    {
        const label base = static_cast<label>(pts.size());
        for (label k = 0; k <= nz; ++k)
            for (label s = 0; s < N; ++s)
                for (label ri = 0; ri < 2; ++ri)
                {
                    const scalar th = scalar(2)*scalar(M_PI)*scalar(s)/scalar(N);
                    const scalar r  = ri ? r1 : r0;
                    pts.push_back(vector{r*std::cos(th), r*std::sin(th), H*scalar(k)/scalar(nz)});
                }
        return base;
    };
    const label baseIn  = addBlock(Nin,  nzA, Rin, R);
    const label baseOut = addBlock(Nout, nzB, R,   Rout);

    auto quad = [&](label a, label b, label c, label d, label o, label n)
    {
        fv.push_back(a); fv.push_back(b); fv.push_back(c); fv.push_back(d);
        foff.push_back(static_cast<label>(fv.size()));
        own.push_back(o);
        if (n >= 0) nei.push_back(n);
    };

    // Windings follow the right-hand rule so Sf points owner->neighbour for internal faces and outward
    // for boundary faces, using the cylindrical triad theta^z = r, z^r = theta, r^theta = z.
    auto emitInternal = [&](label base, label N, label nz, label cell0)
    {
        auto pt   = [&](label ri, label s, label k) { return base + ri + 2*(s % N) + 2*N*k; };
        auto cell = [&](label s, label k) { return cell0 + s + N*k; };
        for (label k = 0; k < nz; ++k)
            for (label s = 0; s < N; ++s)
            {
                const label c = cell(s, k);
                if (s + 1 < N)                        // +theta face at theta_{s+1}
                    quad(pt(0,s+1,k), pt(0,s+1,k+1), pt(1,s+1,k+1), pt(1,s+1,k), c, cell(s+1, k));
                if (s == 0)                           // the wrap face at theta_0, owned by s = 0
                    quad(pt(0,0,k+1), pt(0,0,k), pt(1,0,k), pt(1,0,k+1), c, cell(N-1, k));
                if (k + 1 < nz)                       // +z face at z_{k+1}
                    quad(pt(0,s,k+1), pt(1,s,k+1), pt(1,s+1,k+1), pt(0,s+1,k+1), c, cell(s, k+1));
            }
    };

    auto emitBoundary = [&](label base, label N, label nz, label cell0,
                            const char* innerName, const char* innerType,
                            const char* outerName, const char* outerType,
                            const char* botName, const char* topName)
    {
        auto pt   = [&](label ri, label s, label k) { return base + ri + 2*(s % N) + 2*N*k; };
        auto cell = [&](label s, label k) { return cell0 + s + N*k; };
        auto beginPatch = [&](const char* name, const char* type)
        {
            PatchInfo pi;
            pi.name = name; pi.type = type;
            pi.start = static_cast<label>(own.size());
            patches.push_back(pi);
        };
        auto endPatch = [&]()
        { patches.back().size = static_cast<label>(own.size()) - patches.back().start; };

        beginPatch(innerName, innerType);                       // r = r0, outward -r
        for (label k = 0; k < nz; ++k) for (label s = 0; s < N; ++s)
            quad(pt(0,s,k), pt(0,s,k+1), pt(0,s+1,k+1), pt(0,s+1,k), cell(s,k), -1);
        endPatch();
        beginPatch(outerName, outerType);                       // r = r1, outward +r
        for (label k = 0; k < nz; ++k) for (label s = 0; s < N; ++s)
            quad(pt(1,s,k), pt(1,s+1,k), pt(1,s+1,k+1), pt(1,s,k+1), cell(s,k), -1);
        endPatch();
        beginPatch(botName, "wall");                            // z = 0, outward -z
        for (label s = 0; s < N; ++s)
            quad(pt(0,s,0), pt(0,s+1,0), pt(1,s+1,0), pt(1,s,0), cell(s,0), -1);
        endPatch();
        beginPatch(topName, "wall");                            // z = H, outward +z
        for (label s = 0; s < N; ++s)
            quad(pt(0,s,nz), pt(1,s,nz), pt(1,s+1,nz), pt(0,s+1,nz), cell(s,nz-1), -1);
        endPatch();
    };

    emitInternal(baseIn,  Nin,  nzA, 0);
    emitInternal(baseOut, Nout, nzB, nCin);
    emitBoundary(baseIn,  Nin,  nzA, 0,    "hub",     "wall",      "AMI_in", "cyclicAMI",
                 "inBot",  "inTop");
    emitBoundary(baseOut, Nout, nzB, nCin, "AMI_out", "cyclicAMI", "shroud", "wall",
                 "outBot", "outTop");

    // The pairing. The patches are geometrically coincident, so this is a plain (zero) translational
    // transform -- brae derives the separation from the two patch centroids.
    for (PatchInfo& p : patches)
    {
        if (p.name == "AMI_in")  { p.neighbourPatch = "AMI_out"; p.transform = "translational"; }
        if (p.name == "AMI_out") { p.neighbourPatch = "AMI_in";  p.transform = "translational"; }
    }

    PrimitiveMesh m;
    m.assign(std::move(pts), std::move(fv), std::move(foff), std::move(own), std::move(nei),
             std::move(patches), nC);
    return m;
}

// Exact volume of the fixture: the POLYGONAL (chorded) volume, not the circular one.
inline scalar tubeVolume(label Nin, label Nout, scalar Rin, scalar R, scalar Rout, scalar H)
{
    auto ring = [&](label N, scalar r0, scalar r1)
    { return scalar(0.5)*std::sin(scalar(2)*scalar(M_PI)/scalar(N))*scalar(N)*(r1*r1 - r0*r0)*H; };
    return ring(Nin, Rin, R) + ring(Nout, R, Rout);
}

}   // namespace tubetest
}   // namespace brae
