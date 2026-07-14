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
        if (fvp[pi].type != "cyclicAMI") continue;
        AMIInterface ai;
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

        // project source faces + map+project target faces
        std::vector<std::vector<vec2>> srcP(S.size), tgtP(T.size);
        for (label i = 0; i < S.size; ++i)
        {
            for (const vector& v : facePoly(S.start+i)) srcP[i].push_back(proj(v));
            orientCCW(srcP[i]);
        }
        for (label j = 0; j < T.size; ++j)
        {
            for (const vector& v : facePoly(T.start+j)) tgtP[j].push_back(proj(mapTtoS(v)));
            orientCCW(tgtP[j]);
        }

        // faceAreaWeightAMI: per src face, overlap-area against every tgt face (brute force; OF uses an advancing
        // front, same result). weight = overlap/srcMagSf; weightsSum = coverage fraction.
        ai.ownCell = S.faceCells;
        ai.srcOffset.assign(S.size + 1, 0);
        std::vector<std::vector<std::pair<label,scalar>>> stencil(S.size);   // per src face: (tgtFace j, weight)
        for (label i = 0; i < S.size; ++i)
        {
            const scalar srcArea = g.magSf()[S.start+i];
            for (label j = 0; j < T.size; ++j)
            {
                const scalar ov = overlapArea(srcP[i], tgtP[j]);
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
        out.push_back(std::move(ai));
    }
    return out;
}

} // namespace brae
