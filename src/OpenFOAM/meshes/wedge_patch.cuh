#pragma once
// brae::wedgeGeometry -- OF wedgePolyPatch::calcGeometry (meshes/polyMesh/polyPatches/constraint/wedge).
//
// A `wedge` is how OpenFOAM does AXISYMMETRY: the mesh is one cell thick in the azimuthal direction,
// bounded by two planes making a small angle (~2.5 deg each side) about a coordinate plane, and the two
// planes are coupled by a ROTATION about the symmetry axis. It is a constraint patch, not a boundary
// condition the user chooses: the patch type IS the physics.
//
// The geometry OF derives, and this reproduces:
//
//   n             the patch's average unit face normal
//   centreNormal  the COORDINATE direction the wedge is symmetric about, picked componentwise as
//                 sign(n_i)*(max(|n_i|, 0.5) - 0.5) then normalised -- i.e. the axis n leans against
//   cosAngle      centreNormal & n, the cosine of the HALF angle
//   axis          normalise(centreNormal ^ n), the symmetry axis
//   faceT         rotationTensor(centreNormal, n) -- rotation by the HALF angle: cell -> this patch
//   cellT         faceT & faceT                   -- rotation by the FULL angle: this plane -> the other
//
// OF refuses two degeneracies outright and so does this: a centre plane that is not a coordinate plane
// (|sum of centreNormal components| < 1), and a wedge plane that lies IN a coordinate plane (zero axis),
// which means the case has no wedge angle at all.
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include "fv_patch.cuh"
#include "cyclic_interface.cuh"   // rotationTensor(axis, angle)
#include <cmath>
#include <stdexcept>
#include <string>

namespace brae {

struct WedgeGeometry
{
    vector n{0, 0, 1};
    vector centreNormal{0, 0, 1};
    vector axis{1, 0, 0};
    scalar cosAngle = 1;
    tensor faceT{1, 0, 0, 0, 1, 0, 0, 0, 1};   // rotation by the half angle  (OF wedgePolyPatch::faceT)
    tensor cellT{1, 0, 0, 0, 1, 0, 0, 0, 1};   // rotation by the full angle  (OF wedgePolyPatch::cellT)
};

// OF transform.H rotationTensor(n1, n2): the rotation taking unit n1 onto unit n2.
inline tensor rotationTensorBetween(const vector& n1, const vector& n2)
{
    const scalar s = dot(n1, n2);
    const vector n3 = cross(n1, n2);
    const scalar m3 = magSqr(n3);
    if (m3 > scalar(1e-30))
    {
        // s*I + (1 - s)*sqr(n3)/magSqr(n3) + (n2*n1 - n1*n2), with sqr(v) = v (x) v
        const tensor sq = outer(n3, n3);
        const tensor a  = outer(n2, n1);
        const tensor b  = outer(n1, n2);
        tensor R{};
        const scalar k = (scalar(1) - s)/m3;
        R.xx = s + k*sq.xx + (a.xx - b.xx);  R.xy = k*sq.xy + (a.xy - b.xy);  R.xz = k*sq.xz + (a.xz - b.xz);
        R.yx = k*sq.yx + (a.yx - b.yx);  R.yy = s + k*sq.yy + (a.yy - b.yy);  R.yz = k*sq.yz + (a.yz - b.yz);
        R.zx = k*sq.zx + (a.zx - b.zx);  R.zy = k*sq.zy + (a.zy - b.zy);  R.zz = s + k*sq.zz + (a.zz - b.zz);
        return R;
    }
    if (s < scalar(0))   // contradirectional: OF returns the mirror I + 2*n1*n2
    {
        const tensor m = outer(n1, n2);
        return {scalar(1) + 2*m.xx, 2*m.xy, 2*m.xz,
                2*m.yx, scalar(1) + 2*m.yy, 2*m.yz,
                2*m.zx, 2*m.zy, scalar(1) + 2*m.zz};
    }
    return {1, 0, 0, 0, 1, 0, 0, 0, 1};
}

inline tensor tensorProduct(const tensor& a, const tensor& b)   // a & b
{
    return {a.xx*b.xx + a.xy*b.yx + a.xz*b.zx,  a.xx*b.xy + a.xy*b.yy + a.xz*b.zy,  a.xx*b.xz + a.xy*b.yz + a.xz*b.zz,
            a.yx*b.xx + a.yy*b.yx + a.yz*b.zx,  a.yx*b.xy + a.yy*b.yy + a.yz*b.zy,  a.yx*b.xz + a.yy*b.yz + a.yz*b.zz,
            a.zx*b.xx + a.zy*b.yx + a.zz*b.zx,  a.zx*b.xy + a.zy*b.yy + a.zz*b.zy,  a.zx*b.xz + a.zy*b.yz + a.zz*b.zz};
}

// Built from the patch's own unit face normals (FvPatch::nf), so it needs no more than the patch --
// which is what lets the ordinary patch-field factory construct a wedge without a geometry argument.
inline WedgeGeometry wedgeGeometry(const FvPatch& p)
{
    if (p.size <= 0)
        throw std::runtime_error("brae: wedge patch '" + p.name + "' has no faces.");
    WedgeGeometry w;
    vector s{0, 0, 0};
    for (label i = 0; i < p.size; ++i)
    {
        s.x += p.nf[i].x;  s.y += p.nf[i].y;  s.z += p.nf[i].z;
    }
    const scalar ms = mag(s);
    if (!(ms > scalar(0)))
        throw std::runtime_error("brae: wedge patch '" + p.name + "' has a degenerate average normal.");
    w.n = vector{s.x/ms, s.y/ms, s.z/ms};

    auto sgn = [](scalar v) { return v < scalar(0) ? scalar(-1) : scalar(1); };
    w.centreNormal = vector{sgn(w.n.x)*(std::max(std::fabs(w.n.x), scalar(0.5)) - scalar(0.5)),
                            sgn(w.n.y)*(std::max(std::fabs(w.n.y), scalar(0.5)) - scalar(0.5)),
                            sgn(w.n.z)*(std::max(std::fabs(w.n.z), scalar(0.5)) - scalar(0.5))};
    const scalar mc = mag(w.centreNormal);
    if (!(mc > scalar(0)))
        throw std::runtime_error(
            "brae: wedge patch '" + p.name + "' has no dominant normal direction; its centre plane "
            "cannot be identified. OpenFOAM requires the wedge planes to straddle a COORDINATE plane.");
    w.centreNormal = vector{w.centreNormal.x/mc, w.centreNormal.y/mc, w.centreNormal.z/mc};

    const scalar cnSum = w.centreNormal.x + w.centreNormal.y + w.centreNormal.z;
    if (std::fabs(cnSum) < scalar(1) - scalar(1e-6))
        throw std::runtime_error(
            "brae: wedge patch '" + p.name + "' does not straddle a coordinate plane (its implied centre "
            "normal is not a coordinate direction). OpenFOAM refuses the same geometry.");

    w.cosAngle = dot(w.centreNormal, w.n);
    const vector ax = cross(w.centreNormal, w.n);
    const scalar ma = mag(ax);
    if (!(ma > scalar(1e-12)))
        throw std::runtime_error(
            "brae: wedge patch '" + p.name + "' lies IN a coordinate plane -- it has no wedge angle, so "
            "there is no rotation to apply. The two wedge planes should make a small angle (~2.5 deg) "
            "either side of the plane.");
    w.axis  = vector{ax.x/ma, ax.y/ma, ax.z/ma};
    w.faceT = rotationTensorBetween(w.centreNormal, w.n);
    w.cellT = tensorProduct(w.faceT, w.faceT);
    return w;
}

} // namespace brae
