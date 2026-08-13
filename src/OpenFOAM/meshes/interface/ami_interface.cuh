#pragma once
// brae::AMIInterface, the cyclicAMI (Arbitrary Mesh Interface) lduInterface. Unlike cyclic (1:1 face pairing),
// the two patches are NON-CONFORMING: each SOURCE face couples to MULTIPLE neighbour cells via an area-weighted
// interpolation. Mirrors OpenFOAM AMIInterpolation (faceAreaWeightAMI) + cyclicAMIPolyPatch / cyclicAMIFvPatchField.
//
// Coupling (OF cyclicAMIFvPatchField::updateInterfaceMatrix / patchNeighbourField):
//   pnf[srcFace] = sum_j srcWeights[srcFace][j] * transform(forwardT, psi[ tgtCell[srcFace][j] ])
//   result[ownCell[srcFace]] -= coeff[srcFace] * pnf[srcFace]
// i.e. the single cyclic (own,nbr) pair becomes a weighted STENCIL (own, list of (tgtCell, weight)).
//
// Weights (faceAreaWeightAMI): map the target faces to the source side via the transform, project both to the
// source patch's average plane, and clip each src/tgt polygon pair (Sutherland-Hodgman) -> overlap area.
//   srcWeights[i][j] = overlap_ij / srcMagSf[i]    (conformal=false normalisation; sum_j = coverage fraction)
//   srcWeightsSum[i] = sum_j overlap_ij / srcMagSf[i]
// (this header: host build + weights; device coupling is in device_ami.{cuh,cu}). Both sides are built (symmetric).
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "cf_types.cuh"
#include "interface/cyclic_interface.cuh"   // rotationTensor + transform conventions
#include <map>
#include <stdexcept>
#include <string>
#include <cstdio>
#include <vector>
#include <cmath>
#include <algorithm>
#include <utility>

namespace brae {

struct AMIInterface
{
    label patch = -1, nbrPatch = -1;            // this (source) patch and its paired (target) patch
    std::vector<label>  ownCell;                // owner cell of each SOURCE face (size = nSrc)
    std::vector<label>  srcOffset;              // CSR offsets into (nbrCell, weight), size nSrc+1
    std::vector<label>  nbrCell;                // target-face owner cell per stencil entry
    std::vector<scalar> weight;                 // normalised AMI weight per stencil entry
    std::vector<scalar> weightsSum;             // per src face: coverage = sum of weights (pre-lowWeight)
    std::vector<scalar> magSf;                  // per src face |Sf|
    std::vector<vector> Sf;                     // per src face area vector (out of ownCell)
    std::vector<scalar> deltaCoeffs;            // per src face 1/(nf & delta), delta to the AMI-interpolated nbr
    std::vector<scalar> weights;                // per src face interp weight: face = w*own + (1-w)*nbr_interp
    std::vector<vector> corrVec;                // per src face non-orth correction vector (nf - delta*deltaCoeffs)
    std::vector<vector> dOwn;                   // per src face Cf - C[own] (linearUpwind own-side reconstruction delta)
    std::vector<vector> dNbr;                   // per STENCIL ENTRY Cf_tgt - C[nbr] (UN-rotated, nbr-side reconstruction)
    // cyclicACMI. The interface is only PARTIALLY coupled: it shares its faces with a coincident wall
    // (nonOverlapPatch), and the overlap fraction decides how the area splits between them. Partial
    // coverage is the DEFINING FEATURE here, not a defect -- so ACMI is exempt from the coverage
    // refusal below, and carries no transform (see `separation`).
    bool                acmi = false;
    std::vector<scalar> mask;                   // OF cyclicACMIPolyPatch.C:348, clamp(weightsSum, 0, 1)
    bool                translational = true;
    vector              separation{0,0,0};      // translational period (Cf_nbr - Cf_own at a matched point)
    tensor              forwardT{1,0,0,0,1,0,0,0,1};   // rotational nbr->own rotation (identity if translational)
};

namespace ami_detail {
struct vec2 { scalar x, y; };
inline scalar signedArea(const std::vector<vec2>& p)
{
    scalar a = 0;
    const int n = (int)p.size();
    for (int i = 0; i < n; ++i)
    {
        const vec2 &u = p[i], &v = p[(i+1)%n];
        a += u.x*v.y - v.x*u.y;
    }
    return 0.5*a;
}
inline void orientCCW(std::vector<vec2>& p) { if (signedArea(p) < 0) std::reverse(p.begin(), p.end()); }
inline scalar cross2(const vec2& a, const vec2& b, const vec2& c)
{
    return (b.x-a.x)*(c.y-a.y) - (b.y-a.y)*(c.x-a.x);
}
// Sutherland-Hodgman: clip `subject` (CCW) by convex `clip` (CCW). Returns the clipped polygon (CCW).
inline std::vector<vec2> clipPoly(const std::vector<vec2>& subject, const std::vector<vec2>& clip)
{
    std::vector<vec2> out = subject;
    const int nc = (int)clip.size();
    for (int e = 0; e < nc && !out.empty(); ++e)
    {
        const vec2 A = clip[e], B = clip[(e+1)%nc];                 // clip edge A->B (inside = left, CCW)
        const std::vector<vec2> in = out;
        out.clear();
        const int ni = (int)in.size();
        for (int i = 0; i < ni; ++i)
        {
            const vec2 P = in[i], Q = in[(i+1)%ni];
            const scalar sp = cross2(A,B,P), sq = cross2(A,B,Q);
            if (sp >= 0) out.push_back(P);
            if ((sp >= 0) != (sq >= 0))                            // edge P->Q crosses the clip line
            {
                const scalar t = sp / (sp - sq);
                out.push_back({P.x + t*(Q.x-P.x), P.y + t*(Q.y-P.y)});
            }
        }
    }
    return out;
}
inline scalar overlapArea(const std::vector<vec2>& a, const std::vector<vec2>& b)
{
    const std::vector<vec2> c = clipPoly(a, b);
    return c.size() < 3 ? 0.0 : std::fabs(signedArea(c));
}
} // namespace ami_detail

// Build the cyclicAMI interfaces (both source+target sides) with faceAreaWeightAMI weights.
//
// WEIGHTS ARE NORMALISED BY g.rawMagSf(), NOT g.magSf(). They differ only on a cyclicACMI mesh, where
// the coupled area has already been scaled by the overlap mask: normalising the overlap by the scaled
// area would return 1 for every face and destroy the very mask it expresses. OF has the same separation
// for free -- its AMI is built on the primitivePatch,
// whose areas come straight from the point positions, while the SCALED areas live on the polyPatch
// ("using primitivePatch face areas since these are based on the raw point locations (not affected by
// ACMI scaling)", cyclicACMIPolyPatch.C:394).
//
// The flux areas stored on the interface (ai.Sf / ai.magSf) deliberately keep using g, i.e. the SCALED
// values -- those are the areas the interface actually transports through.
//
// NOTE ON NORMALISATION, verified against OF's own log. OF normalises AMI weights two ways
// (AMIInterpolation.C:159-208), chosen by `conformal` = requireMatch:
//     requireMatch 1 (cyclicAMI) : denom = sum(overlap)  -> weights sum to exactly 1
//     requireMatch 0 (cyclicACMI): denom = face area     -> weights sum to the COVERAGE
// and the ACMI polyMesh boundary carries `requireMatch 0`. brae divides by the face area and never
// renormalises, so it is on OF's ACMI branch already -- confirmed by OF's printed weight sums on
// oscillatingInletACMI2D at t = 0.292: average 0.7578655102 over 40 faces with 30 covered / 1 blended /
// 9 uncovered, i.e. (30 + 0.3146)/40. Conformal weights would have given 31/40 = 0.775.
inline std::vector<AMIInterface> buildAMIInterfaces(
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& fvp)
{
    using namespace ami_detail;
    std::map<std::string, label> nameToIdx;
    for (label pi = 0; pi < (label)fvp.size(); ++pi) nameToIdx[fvp[pi].name] = pi;
    const std::vector<PatchInfo>& pinfo = m.patches();
    const std::vector<vector>& pts = m.points();
    const std::vector<label>& fv = m.faceVerts();
    const std::vector<label>& fo = m.faceOffsets();

    // face polygon (global face index) as a list of 3D vertices
    auto facePoly = [&](label gf)
    {
        std::vector<vector> P;
        for (label k = fo[gf]; k < fo[gf+1]; ++k) P.push_back(pts[fv[k]]);
        return P;
    };

    std::vector<AMIInterface> out;
    for (label pi = 0; pi < (label)fvp.size(); ++pi)
    {
        const bool isACMI = (fvp[pi].type == "cyclicACMI");
        if (fvp[pi].type != "cyclicAMI" && !isACMI) continue;
        AMIInterface ai;
        ai.acmi = isACMI;
        ai.patch = pi;
        const std::string nbrName = pinfo[pi].neighbourPatch;
        const auto it = nameToIdx.find(nbrName);
        if (it == nameToIdx.end()) throw std::runtime_error("cyclicAMI: neighbourPatch '" + nbrName + "' not found");
        ai.nbrPatch = it->second;
        ai.translational = (pinfo[pi].transform != "rotational");
        const FvPatch& S = fvp[pi];
        const FvPatch& T = fvp[ai.nbrPatch];

        // transform (nbr/target -> own/source); same convention as cyclic
        vector axis{0,0,1};
        vector ctr{0,0,0};
        if (!ai.translational)
        {
            axis = pinfo[pi].rotationAxis / mag(pinfo[pi].rotationAxis);
            ctr = pinfo[pi].rotationCentre;
            const vector po = g.Cf()[S.start] - ctr, pn = g.Cf()[T.start] - ctr;
            const vector perpO = po - dot(po,axis)*axis, perpN = pn - dot(pn,axis)*axis;
            const scalar angle = std::atan2(dot(cross(perpN, perpO), axis), dot(perpN, perpO));
            ai.forwardT = rotationTensor(axis, angle);
        }
        // translational period = mean(Cf_tgt) - mean(Cf_src). Robust for NON-conforming patches (faces are not
        // ordered to match, so a face-0 difference is wrong): = the period vector for translational periodicity, ~=0
        // for coincident non-conformal joins. (A dict separationVector would override this; not yet read.)
        // cyclicACMI carries NO transform: the two patches are CO-LOCATED (OF's blockMeshDict gives
        // them the same faces) and any offset between their centroids is the physical slide of the
        // moving zone, not a period. Inferring a period there is catastrophic and silent -- measured on
        // pimpleFoam/RAS/oscillatingInletACMI2D at t=0.5, where the inlet channel has slid 0.5 in y:
        //
        //     inferred separation = (0 -0.5 0)   -> coverage 0 uncovered, 0 blended, 40 covered
        //     separation = 0                     -> coverage 19 uncovered, 1 blended, 20 covered
        //     OF's own report at t=0.5           -> 19, 1, 20
        //
        // With the inferred period the patches are re-aligned, the mask is identically 1 for all time,
        // the blockage wall gets zero area, and the sliding channel NEVER CLOSES -- a case that runs
        // clean and is entirely wrong.
        if (!ai.acmi)
        {
            vector cS{0,0,0}, cT{0,0,0};
            for (label i = 0; i < S.size; ++i) cS += g.Cf()[S.start+i];
            for (label j = 0; j < T.size; ++j) cT += g.Cf()[T.start+j];
            ai.separation = cT/(scalar)T.size - cS/(scalar)S.size;
        }
        // map a target-side point to the source side (so src & mapped-tgt overlap in the same frame)
        auto mapTtoS = [&](const vector& p) -> vector
        {
            if (ai.translational) return p - ai.separation;
            return dot(p - ctr, transpose(ai.forwardT)) + ctr;   // forwardT*(p-ctr)+ctr  (nbr->own rotation)
        };

        // average source-plane basis (e1,e2,n): n = mean source unit normal
        vector n{0,0,0};
        for (label i = 0; i < S.size; ++i) n += g.Sf()[S.start+i] / g.magSf()[S.start+i];
        n = n / std::fmax(mag(n), 1e-300);
        vector e1 = std::fabs(n.x) < 0.9 ? cross(n, vector{1,0,0}) : cross(n, vector{0,1,0});
        e1 = e1 / std::fmax(mag(e1), 1e-300);
        vector e2 = cross(n, e1);
        const vector orig = g.Cf()[S.start];
        auto proj = [&](const vector& p) -> vec2
        {
            const vector d = p - orig;
            return { dot(d,e1), dot(d,e2) };
        };

        // 3-D polygons, kept unprojected: OF picks the projection direction PER SOURCE/TARGET PAIR,
        // so there is no one plane to flatten onto up front.
        std::vector<std::vector<vector>> srcW(S.size), tgtW(T.size);
        for (label i = 0; i < S.size; ++i)
            for (const vector& v : facePoly(S.start+i)) srcW[i].push_back(v);
        for (label j = 0; j < T.size; ++j)
            for (const vector& v : facePoly(T.start+j)) tgtW[j].push_back(mapTtoS(v));

        // Unit normals on the source side (the target's mapped to it), for the pair normal below.
        auto unitN = [&](label f) {
            const vector& Sf = g.Sf()[f]; const scalar a = g.magSf()[f];
            return a > 0 ? vector{Sf.x/a, Sf.y/a, Sf.z/a} : vector{0,0,0};
        };
        std::vector<vector> nSrc(S.size), nTgt(T.size);
        for (label i = 0; i < S.size; ++i) nSrc[i] = unitN(S.start+i);
        // A normal maps like a DIRECTION, not a point: unchanged under translation, rotated under a
        // rotational transform (the same forwardT the delta uses below).
        for (label j = 0; j < T.size; ++j)
        {
            const vector nj = unitN(T.start+j);
            nTgt[j] = ai.translational ? nj : dot(nj, transpose(ai.forwardT));
        }

        // OF faceAreaWeightAMI::calcInterArea (faceAreaWeightAMI.C:402-410):
        //     n = -srcNormal (+/-) tgtNormal ;  project the pair along n/|n|
        // A projection direction per PAIR, not one average plane for the whole patch. That is what
        // makes a curved interface work: each pair is locally planar even when the patch is a
        // cylinder, whose average normal nearly cancels. brae used a single source-patch average
        // plane, which collapsed opposite sides of the cylinder onto each other and lost 55% of the
        // face coverage on pimpleFoam/RAS/rotatingFanInRoom.
        auto projectPair = [&](const std::vector<vector>& poly, const vector& e1, const vector& e2,
                               const vector& orig)
        {
            std::vector<vec2> out;
            out.reserve(poly.size());
            for (const vector& v : poly)
            {
                const vector d{v.x-orig.x, v.y-orig.y, v.z-orig.z};
                out.push_back({ dot(d,e1), dot(d,e2) });
            }
            orientCCW(out);
            return out;
        };

        // faceAreaWeightAMI: per src face, overlap-area against every tgt face (brute force; OF uses an advancing
        // front, same result). weight = overlap/srcMagSf; weightsSum = coverage fraction.
        ai.ownCell = S.faceCells;
        ai.srcOffset.assign(S.size + 1, 0);
        std::vector<std::vector<std::pair<label,scalar>>> stencil(S.size);   // per src face: (tgtFace j, weight)
        for (label i = 0; i < S.size; ++i)
        {
            const scalar srcArea = g.rawMagSf(S.start+i);   // RAW: see the note on the signature
            for (label j = 0; j < T.size; ++j)
            {
                // per-pair projection normal (OF: -nSrc + nTgt, reversed target subtracts)
                // OF: n = -srcNormal + tgtNormal. The two AMI patches face EACH OTHER, so the mapped
                // target normal opposes the source one and the sum is ~ -2*nSrc -- a well-defined
                // direction. It degenerates only if the two point the same way, which the magN guard
                // below rejects rather than projecting onto a near-zero direction.
                vector n{-nSrc[i].x + nTgt[j].x, -nSrc[i].y + nTgt[j].y, -nSrc[i].z + nTgt[j].z};
                const scalar magN = std::sqrt(n.x*n.x + n.y*n.y + n.z*n.z);
                if (magN <= 1e-150) continue;             // OF: ROOTVSMALL -> skip the pair
                n = vector{n.x/magN, n.y/magN, n.z/magN};

                // an orthonormal basis in the plane perpendicular to n
                const vector a = (std::fabs(n.x) < 0.9) ? vector{1,0,0} : vector{0,1,0};
                vector e1{a.y*n.z - a.z*n.y, a.z*n.x - a.x*n.z, a.x*n.y - a.y*n.x};
                const scalar m1 = std::sqrt(e1.x*e1.x + e1.y*e1.y + e1.z*e1.z);
                if (m1 <= 1e-150) continue;
                e1 = vector{e1.x/m1, e1.y/m1, e1.z/m1};
                const vector e2{n.y*e1.z - n.z*e1.y, n.z*e1.x - n.x*e1.z, n.x*e1.y - n.y*e1.x};

                const vector orig = srcW[i][0];
                const scalar ov = overlapArea(projectPair(srcW[i], e1, e2, orig),
                                              projectPair(tgtW[j], e1, e2, orig));
                if (ov > 1e-14 * srcArea) stencil[i].push_back({ j, ov / srcArea });
            }
            scalar s = 0;
            for (auto& e : stencil[i]) s += e.second;
            ai.weightsSum.push_back(s);
            ai.srcOffset[i+1] = ai.srcOffset[i] + (label)stencil[i].size();
        }
        // per-source-face geometry: the neighbour delta is AMI-interpolated (OF cyclicAMIFvPatch::makeDeltaCoeffs).
        // transform of a nbr DELTA to the src side: identity (translational) or forwardT (rotational).
        auto rotDelta = [&](const vector& d) -> vector
        {
            return ai.translational ? d : dot(d, transpose(ai.forwardT));
        };
        for (label i = 0; i < S.size; ++i)
        {
            const vector Sfi = g.Sf()[S.start+i];
            const scalar msf = g.magSf()[S.start+i];
            const vector nf = Sfi/msf;
            const vector patchD = g.Cf()[S.start+i] - g.C()[S.faceCells[i]];
            vector nbrDint{0,0,0};
            scalar dni = 0;
            for (const auto& e : stencil[i])
            {
                const label j = e.first;
                const vector nbrD = g.Cf()[T.start+j] - g.C()[T.faceCells[j]];
                const vector nfn  = g.Sf()[T.start+j] / g.magSf()[T.start+j];     // neighbour-side unit normal
                nbrDint += e.second * rotDelta(nbrD);                             // Sum w . transform(nbrD)  (for delta)
                dni     += e.second * dot(nfn, nbrD);                             // Sum w . (nfn & nbrD)  (OF makeWeights)
            }
            const vector delta = patchD - nbrDint;
            const scalar di = dot(nf, patchD);
            ai.magSf.push_back(msf);
            ai.Sf.push_back(Sfi);
            ai.weights.push_back(std::fabs(di + dni) > 1e-300 ? dni/(di + dni) : 0.5);
            const scalar dc = 1.0 / std::fmax(dot(nf, delta), 0.05 * mag(delta));
            ai.deltaCoeffs.push_back(dc);
            ai.corrVec.push_back(nf - delta * dc);
            ai.dOwn.push_back(patchD);   // own-side linearUpwind reconstruction delta (Cf - C_own)
            for (const auto& e : stencil[i])
            {
                const label j = e.first;
                ai.nbrCell.push_back(T.faceCells[j]);
                ai.weight.push_back(e.second);
                ai.dNbr.push_back(g.Cf()[T.start+j] - g.C()[T.faceCells[j]]);   // nbr-side delta (Cf_tgt - C_nbr), un-rotated
            }
        }
        // THE ACMI MASK -- OF cyclicACMIPolyPatch.C:348, srcMask_ = clamp(AMI.srcWeightsSum(), 0, 1).
        // The coverage fraction IS the mask: it decides how each face's area splits between the coupled
        // patch and its coincident nonOverlapPatch wall. Computed for every interface (it is just the
        // clamped coverage) but only meaningful, and only used, for ACMI.
        ai.mask.reserve(ai.weightsSum.size());
        for (const scalar w : ai.weightsSum)
            ai.mask.push_back(w < scalar(0) ? scalar(0) : (w > scalar(1) ? scalar(1) : w));

        // COVERAGE CHECK -- for cyclicAMI ONLY.
        //
        // A cyclicAMI source face must be FULLY covered by target faces (weightsSum ~ 1) or it loses
        // that fraction of its flux across the interface. That is a mass sink: continuity grows every
        // step and the run diverges. Measured on pimpleFoam/RAS/rotatingFanInRoom, on a STATIC mesh,
        // back when brae projected both patches onto the source patch's average plane:
        //
        //     planarity |sum Sf|/sum|Sf| = 0.4206      (1.0 = planar)
        //     mean coverage weightsSum   = 0.4492      1481 of 14080 faces at ~0
        //
        // That projection has since been replaced by OF's per-source/target-pair one
        // (faceAreaWeightAMI.C:402-410), which took the same case to mean coverage 1.0001. The check
        // stays as the guard that would catch any future regression of the same kind.
        //
        // NOT APPLIED TO cyclicACMI. There, partial coverage is the entire point: a face that has slid
        // off its neighbour is SUPPOSED to read ~0 and hand its area to the nonOverlapPatch wall. OF's
        // own report on oscillatingInletACMI2D at t=0.5 is 19 uncovered, 1 blended, 20 covered -- mean
        // coverage ~0.5, which this check would refuse outright. Refusing a correct ACMI interface for
        // looking like a broken AMI one would be exactly backwards.
        //
        // REFUSED, not warned. brae was building unusable weights and running to a confident wrong
        // answer, which is the one outcome this codebase does not accept.
        if (!ai.acmi)
        {
            const std::size_t n = ai.weightsSum.size();
            if (n)
            {
                scalar sum = 0, lo = ai.weightsSum[0];
                std::size_t under = 0;
                for (const scalar w : ai.weightsSum)
                {
                    sum += w;
                    lo = std::min(lo, w);
                    if (w < scalar(0.99)) ++under;
                }
                const scalar mean = sum/static_cast<scalar>(n);
                // 0.99 would reject on round-off; 0.9 still catches the 0.45 failure by a wide margin
                // while leaving a genuinely conforming interface alone.
                if (mean < scalar(0.9))
                {
                    char buf[512];
                    std::snprintf(buf, sizeof(buf),
                        "brae: cyclicAMI '%s' has mean face coverage %.4f (min %.4f); %zu of %zu source "
                        "faces are less than 99%% covered, so that fraction of their flux is lost across "
                        "the interface -- a mass sink that grows the continuity error every step until the "
                        "run diverges. Refused rather than solved with a leaking interface. (A partially "
                        "overlapping interface is what cyclicACMI is for, and is not refused.)",
                        S.name.c_str(), (double)mean, (double)lo, under, n);
                    throw std::runtime_error(buf);
                }
            }
        }
        out.push_back(std::move(ai));
    }
    return out;
}

} // namespace brae
