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

std::vector<vector> linearUpwindVFaceCorrection(
    const std::vector<scalar>&    phi,
    const GeometricField<vector>& vf,
    const std::vector<tensor>&    gradVf,
    const PrimitiveMesh&          m,
    const FvGeometry&             g)
{
    const label nIf = m.nInternalFaces();
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();
    const std::vector<scalar>& w  = g.weights();
    const std::vector<vector>& C  = g.C();
    const std::vector<vector>& Cf = g.Cf();
    constexpr scalar VSMALL = 1.0e-300;

    std::vector<vector> out(nIf);
    for (label f = 0; f < nIf; ++f)
    {
        const label P = own[f], N = nei[f];
        const bool  outflow = (phi[f] > 0.0);
        const label up = outflow ? P : N;
        const vector d { Cf[f].x - C[up].x, Cf[f].y - C[up].y, Cf[f].z - C[up].z };
        const tensor& gc = gradVf[up];
        // (d & gradVf)_j = d_i * gradVf_ij, under OpenFOAM's grad(U)_ij = d(U_j)/d(x_i)
        vector corr { d.x*gc.xx + d.y*gc.yx + d.z*gc.zx,
                      d.x*gc.xy + d.y*gc.yy + d.z*gc.zy,
                      d.x*gc.xz + d.y*gc.yz + d.z*gc.zz };

        const vector& vP = vf.internal[P];
        const vector& vN = vf.internal[N];
        const scalar  a  = outflow ? (1.0 - w[f]) : w[f];
        const vector  maxCorr = outflow
            ? vector{ a*(vN.x - vP.x), a*(vN.y - vP.y), a*(vN.z - vP.z) }
            : vector{ a*(vP.x - vN.x), a*(vP.y - vN.y), a*(vP.z - vN.z) };

        const scalar sq = corr.x*corr.x + corr.y*corr.y + corr.z*corr.z;
        const scalar mx = corr.x*maxCorr.x + corr.y*maxCorr.y + corr.z*maxCorr.z;
        if (sq > 0.0)
        {
            if (mx < 0.0)     corr = vector{0, 0, 0};
            else if (sq > mx) { const scalar sc = mx/(sq + VSMALL);
                                corr = vector{corr.x*sc, corr.y*sc, corr.z*sc}; }
        }
        out[f] = corr;
    }
    return out;
}


std::vector<vector> linearUpwindVCorrection(
    const std::vector<scalar>&    phi,
    const GeometricField<vector>& vf,
    const std::vector<tensor>&    gradVf,
    const PrimitiveMesh&          m,
    const FvGeometry&             g)
{
    // faceSum(phi_f * corr_f), computed FROM the face corrections so the two cannot drift apart.
    const std::vector<vector> corr = linearUpwindVFaceCorrection(phi, vf, gradVf, m, g);
    const label nIf = m.nInternalFaces();
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();
    std::vector<vector> src(m.nCells(), vector{0, 0, 0});
    for (label f = 0; f < nIf; ++f)
    {
        const vector fc { phi[f]*corr[f].x, phi[f]*corr[f].y, phi[f]*corr[f].z };
        src[own[f]] += fc;
        src[nei[f]] -= fc;
    }
    return src;
}

} // namespace limitedSchemes
} // namespace cpu
} // namespace brae
