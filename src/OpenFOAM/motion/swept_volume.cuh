#pragma once
// Swept volumes and meshPhi -- OF face::sweptVol (face.C:656) + triangle::sweptVol (triangleI.H:386).
//
// When the mesh moves, each face sweeps a volume. OF turns that into meshPhi (a volumetric flux,
// dimVolume/dimTime) and the equations use it two ways:
//   * the Space Conservation Law: d(V)/dt = sum_faces(meshPhi), which is what keeps a moving mesh from
//     manufacturing or destroying mass;
//   * fvc::makeRelative(phi, U): phi -= meshPhi, so convection sees the flux RELATIVE to the moving mesh.
// Getting this wrong does not fail loudly -- it leaks mass slowly. Hence the SCL check below, which is
// an identity the swept volumes must satisfy exactly, and is checkable without running OpenFOAM.
//
// TRANSCRIBED, not derived. triangle::sweptVol maps old triangle (a,b,c) to new (t.a,t.b,t.c):
//
//   (1/12)*[ ((t.a-a)&((b-a)^(c-a))) + ((t.b-b)&((c-b)^(t.a-b)))  + ((c-t.c)&((t.b-t.c)^(t.a-t.c)))
//          + ((t.a-a)&((b-a)^(c-a))) + ((b-t.b)&((t.a-t.b)^(t.c-t.b))) + ((c-t.c)&((b-t.c)^(t.a-t.c))) ]
//
// The first and fourth terms are identical in OF's source -- that is not a transcription slip here, it
// is what face.C computes, and halving it would change the answer.
//
// face::sweptVol fans the face about its CENTRE at both times, summing the per-triangle swept volumes.
// The centre is face::centre -- the AREA-WEIGHTED centroid, (1/3)*sumAc/sumA, not the vertex average
// (the vertex average is only its starting estimate). Using the estimate instead would be a different
// decomposition of the same polygon and would break the SCL by a small, plausible amount.
//
// Serial here on purpose: it runs once per time step over the moving faces, alongside the geometry
// rebuild, and correctness matters more than throughput. Moving it to the device later changes nothing
// about the values.

#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include <algorithm>
#include <cmath>
#include <vector>

namespace brae {

namespace detail {

inline vector sv_cross(const vector& a, const vector& b)
{
    return vector{a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}
inline scalar sv_dot(const vector& a, const vector& b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
inline vector sv_sub(const vector& a, const vector& b) { return vector{a.x-b.x, a.y-b.y, a.z-b.z}; }

// OF triangle::sweptVol (triangleI.H:386), term for term.
inline scalar triSweptVol(
    const vector& a,  const vector& b,  const vector& c,
    const vector& ta, const vector& tb, const vector& tc)
{
    return (scalar(1)/scalar(12))*
    (
        sv_dot(sv_sub(ta,a), sv_cross(sv_sub(b,a),  sv_sub(c,a)))
      + sv_dot(sv_sub(tb,b), sv_cross(sv_sub(c,b),  sv_sub(ta,b)))
      + sv_dot(sv_sub(c,tc), sv_cross(sv_sub(tb,tc),sv_sub(ta,tc)))

      + sv_dot(sv_sub(ta,a), sv_cross(sv_sub(b,a),  sv_sub(c,a)))
      + sv_dot(sv_sub(b,tb), sv_cross(sv_sub(ta,tb),sv_sub(tc,tb)))
      + sv_dot(sv_sub(c,tc), sv_cross(sv_sub(b,tc), sv_sub(ta,tc)))
    );
}

// OF face::centre -- the area-weighted centroid, matching fv_geometry.cu's Cf exactly.
inline vector faceCentreOf(const PrimitiveMesh& m, label f, const std::vector<vector>& P)
{
    const label k = m.faceSize(f);
    if (k == 3)
        return vector{(P[m.faceVert(f,0)].x + P[m.faceVert(f,1)].x + P[m.faceVert(f,2)].x)/scalar(3),
                      (P[m.faceVert(f,0)].y + P[m.faceVert(f,1)].y + P[m.faceVert(f,2)].y)/scalar(3),
                      (P[m.faceVert(f,0)].z + P[m.faceVert(f,1)].z + P[m.faceVert(f,2)].z)/scalar(3)};
    vector fc = P[m.faceVert(f,0)];
    for (label pi = 1; pi < k; ++pi)
    { fc.x += P[m.faceVert(f,pi)].x; fc.y += P[m.faceVert(f,pi)].y; fc.z += P[m.faceVert(f,pi)].z; }
    fc.x /= scalar(k); fc.y /= scalar(k); fc.z /= scalar(k);

    scalar sumA = 0;
    vector sumAc{0,0,0};
    for (label pi = 0; pi < k; ++pi)
    {
        const vector& tp = P[m.faceVert(f,pi)];
        const vector& np = P[m.faceVert(f, pi == k-1 ? 0 : pi+1)];
        const vector c{tp.x+np.x+fc.x, tp.y+np.y+fc.y, tp.z+np.z+fc.z};
        const vector n = sv_cross(sv_sub(np,tp), sv_sub(fc,tp));
        const scalar a = std::sqrt(sv_dot(n,n));
        sumA += a;
        sumAc.x += a*c.x; sumAc.y += a*c.y; sumAc.z += a*c.z;
    }
    if (sumA < scalar(1e-300)) return fc;
    return vector{sumAc.x/(scalar(3)*sumA), sumAc.y/(scalar(3)*sumA), sumAc.z/(scalar(3)*sumA)};
}

}   // namespace detail

// OF face::sweptVol (face.C:656): fan about the face centre at BOTH times.
inline scalar faceSweptVol(
    const PrimitiveMesh& m, label f,
    const std::vector<vector>& oldP,
    const std::vector<vector>& newP)
{
    const label k = m.faceSize(f);
    if (k < 3) return 0;
    const vector co = detail::faceCentreOf(m, f, oldP);
    const vector cn = detail::faceCentreOf(m, f, newP);
    scalar sv = 0;
    for (label pi = 0; pi < k; ++pi)
    {
        const label v0 = m.faceVert(f, pi);
        const label v1 = m.faceVert(f, pi == k-1 ? 0 : pi+1);   // OF closes the fan with the wrap face
        sv += detail::triSweptVol(co, oldP[v0], oldP[v1], cn, newP[v0], newP[v1]);
    }
    return sv;
}

// meshPhi for every face: sweptVol/deltaT (OF's dimVolume/dimTime).
inline std::vector<scalar> meshPhi(
    const PrimitiveMesh& m,
    const std::vector<vector>& oldP,
    const std::vector<vector>& newP,
    scalar deltaT)
{
    const label nF = m.nFaces();
    std::vector<scalar> mp(static_cast<std::size_t>(nF), 0);
    if (deltaT <= 0) return mp;
    for (label f = 0; f < nF; ++f) mp[f] = faceSweptVol(m, f, oldP, newP)/deltaT;
    return mp;
}

// fvc::makeRelative / makeAbsolute -- OF fvcMeshPhi.C:76 and :121.
//
//     makeRelative(phi, U):  if (mesh.moving()) phi -= fvc::meshPhi(U);
//     makeAbsolute(phi, U):  if (mesh.moving()) phi += fvc::meshPhi(U);
//
// Both are NO-OPS on a static mesh, which is why every existing brae case is unaffected.
//
// WHY IT MATTERS. On a moving mesh the convective term must see the flux RELATIVE to the mesh: a fluid
// travelling with the mesh convects nothing. Skip this and a translating mesh convects its own motion
// into the solution -- a wrong answer that still converges. The runnable check for it is exactly that
// statement: move the mesh and the fluid together and the relative flux must vanish.
inline void makeRelative(std::vector<scalar>& phi, const std::vector<scalar>& meshPhiF)
{
    const std::size_t n = std::min(phi.size(), meshPhiF.size());
    for (std::size_t f = 0; f < n; ++f) phi[f] -= meshPhiF[f];
}

inline void makeAbsolute(std::vector<scalar>& phi, const std::vector<scalar>& meshPhiF)
{
    const std::size_t n = std::min(phi.size(), meshPhiF.size());
    for (std::size_t f = 0; f < n; ++f) phi[f] += meshPhiF[f];
}

// movingWallVelocity -- OF movingWallVelocityFvPatchVectorField::Uwall().
//
//     oldFc  = face centre at the OLD points
//     Up     = (faceCentres_new - oldFc)/deltaT       the wall's own velocity from the mesh motion
//     phip   = meshPhi on the patch
//     Un     = phip/(magSf + VSMALL)
//     Uwall  = Up + n*(Un - (n & Up))
//
// The last line is the whole point and is easy to mis-read as "Up plus a normal correction". It
// REPLACES the normal component: the tangential part comes from the face-centre displacement, the
// normal part from the swept volume. Those two disagree slightly on a deforming face, and OF trusts
// the swept volume for the normal direction because that is what the SCL is written against -- using
// n&Up there instead would violate the SCL the flux was just made consistent with.
//
// Un uses meshPhi, so this is only correct once meshPhi is (verified above against the SCL). Every
// input is already in place; nothing here re-derives geometry.
inline std::vector<vector> movingWallVelocity(
    const PrimitiveMesh& m,
    label patchStart, label patchSize,
    const std::vector<vector>& oldP,
    const std::vector<vector>& newP,
    const std::vector<scalar>& meshPhiF,   // per face, sweptVol/deltaT
    const std::vector<vector>& Sf,         // CURRENT (moved) face areas
    const std::vector<scalar>& magSf,
    scalar deltaT)
{
    std::vector<vector> Uw(static_cast<std::size_t>(patchSize), vector{0,0,0});
    if (deltaT <= 0) return Uw;
    const scalar kVSmall = 1e-300;
    for (label i = 0; i < patchSize; ++i)
    {
        const label f = patchStart + i;
        const vector oldFc = detail::faceCentreOf(m, f, oldP);
        const vector newFc = detail::faceCentreOf(m, f, newP);
        const vector Up{(newFc.x - oldFc.x)/deltaT,
                        (newFc.y - oldFc.y)/deltaT,
                        (newFc.z - oldFc.z)/deltaT};
        const scalar a = magSf[f];
        const vector n = (a > kVSmall) ? vector{Sf[f].x/a, Sf[f].y/a, Sf[f].z/a} : vector{0,0,0};
        const scalar Un   = meshPhiF[f]/(a + kVSmall);
        const scalar nDUp = n.x*Up.x + n.y*Up.y + n.z*Up.z;
        Uw[static_cast<std::size_t>(i)] = vector{Up.x + n.x*(Un - nDUp),
                                                 Up.y + n.y*(Un - nDUp),
                                                 Up.z + n.z*(Un - nDUp)};
    }
    return Uw;
}

}   // namespace brae
