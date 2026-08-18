// _cpp REFERENCE implementation -- see limitedSchemes_cpp.cuh for the OpenFOAM provenance.
#include "limitedSchemes_cpp.cuh"
#include <cmath>

namespace brae {
namespace cpu {
namespace limitedSchemes {

namespace {

inline scalar sign0(scalar x) { return (x >= 0) ? 1.0 : -1.0; }   // OpenFOAM sign(): >= 0 -> +1

// NVDTVD::r -- the scalar TVD ratio. `gradf` is the face difference, `gradcf` the projection of the
// UPWIND cell's gradient onto d. The 1000x guard is OpenFOAM's, not a regularisation of our own.
inline scalar rScalar(scalar faceFlux, scalar phiP, scalar phiN,
                      const vector& gradcP, const vector& gradcN, const vector& d)
{
    const scalar gradf = phiN - phiP;
    const vector& gc = (faceFlux > 0) ? gradcP : gradcN;
    const scalar gradcf = d.x*gc.x + d.y*gc.y + d.z*gc.z;
    if (std::fabs(gradcf) >= 1000.0*std::fabs(gradf))
        return 2.0*1000.0*sign0(gradcf)*sign0(gradf) - 1.0;
    return 2.0*(gradcf/gradf) - 1.0;
}

// NVDVTVDV::r -- the V form. gradf is the SQUARED magnitude of the vector difference and gradcf is that
// difference dotted with (d & gradc), so one limiter serves all three components. That coupling is the
// whole difference between `limitedLinear` and `limitedLinearV`.
inline scalar rVector(scalar faceFlux, const vector& phiP, const vector& phiN,
                      const tensor& gradcP, const tensor& gradcN, const vector& d)
{
    const vector gradfV { phiN.x - phiP.x, phiN.y - phiP.y, phiN.z - phiP.z };
    const scalar gradf = gradfV.x*gradfV.x + gradfV.y*gradfV.y + gradfV.z*gradfV.z;
    const tensor& gc = (faceFlux > 0) ? gradcP : gradcN;
    // (d & gradc)_j = d_i * gradc_ij   -- OpenFOAM's grad(U)_ij = d(U_j)/d(x_i)
    const vector dg { d.x*gc.xx + d.y*gc.yx + d.z*gc.zx,
                      d.x*gc.xy + d.y*gc.yy + d.z*gc.zy,
                      d.x*gc.xz + d.y*gc.yz + d.z*gc.zz };
    const scalar gradcf = gradfV.x*dg.x + gradfV.y*dg.y + gradfV.z*dg.z;
    if (std::fabs(gradcf) >= 1000.0*std::fabs(gradf))
        return 2.0*1000.0*sign0(gradcf)*sign0(gradf) - 1.0;
    return 2.0*(gradcf/gradf) - 1.0;
}

inline scalar clamp01(scalar x) { return x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x); }

} // namespace


std::vector<scalar> upwindWeights(const std::vector<scalar>& phi)
{
    std::vector<scalar> w(phi.size());
    for (std::size_t f = 0; f < phi.size(); ++f) w[f] = (phi[f] >= 0.0) ? 1.0 : 0.0;   // pos0
    return w;
}


std::vector<scalar> lustWeights(const std::vector<scalar>& phi, const FvGeometry& g)
{
    const std::vector<scalar>& cd = g.weights();
    std::vector<scalar> w(phi.size());
    for (std::size_t f = 0; f < phi.size(); ++f)
        w[f] = 0.75*cd[f] + 0.25*((phi[f] >= 0.0) ? 1.0 : 0.0);
    return w;
}


std::vector<scalar> limitedLinearWeights(
    const std::vector<scalar>&    phi,
    const GeometricField<scalar>& vf,
    const std::vector<vector>&    gradVf,
    scalar                        k,
    const PrimitiveMesh&          m,
    const FvGeometry&             g)
{
    const label nIf = m.nInternalFaces();
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();
    const std::vector<scalar>& cd = g.weights();
    const std::vector<vector>& C  = g.C();
    const scalar twoByk = 2.0 / std::fmax(k, 1e-15);          // limitedLinear.H:82

    std::vector<scalar> w(nIf);
    for (label f = 0; f < nIf; ++f)
    {
        const label P = own[f], N = nei[f];
        const vector d { C[N].x - C[P].x, C[N].y - C[P].y, C[N].z - C[P].z };
        const scalar r = rScalar(phi[f], vf.internal[P], vf.internal[N], gradVf[P], gradVf[N], d);
        const scalar lim = clamp01(twoByk * r);
        w[f] = lim*cd[f] + (1.0 - lim)*((phi[f] >= 0.0) ? 1.0 : 0.0);
    }
    return w;
}


std::vector<scalar> limitedLinearVWeights(
    const std::vector<scalar>&    phi,
    const GeometricField<vector>& vf,
    const std::vector<tensor>&    gradVf,
    scalar                        k,
    const PrimitiveMesh&          m,
    const FvGeometry&             g)
{
    const label nIf = m.nInternalFaces();
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();
    const std::vector<scalar>& cd = g.weights();
    const std::vector<vector>& C  = g.C();
    const scalar twoByk = 2.0 / std::fmax(k, 1e-15);

    std::vector<scalar> w(nIf);
    for (label f = 0; f < nIf; ++f)
    {
        const label P = own[f], N = nei[f];
        const vector d { C[N].x - C[P].x, C[N].y - C[P].y, C[N].z - C[P].z };
        const scalar r = rVector(phi[f], vf.internal[P], vf.internal[N], gradVf[P], gradVf[N], d);
        const scalar lim = clamp01(twoByk * r);
        w[f] = lim*cd[f] + (1.0 - lim)*((phi[f] >= 0.0) ? 1.0 : 0.0);
    }
    return w;
}

} // namespace limitedSchemes
} // namespace cpu
} // namespace brae
