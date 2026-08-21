#include "cyclicAMI_cpp.cuh"

#include <cmath>

namespace brae {
namespace cpu {
namespace cyclicAMI {

std::vector<scalar> interpolate(const AMIInterface& a, const std::vector<scalar>& psi)
{
    const std::size_t n = a.ownCell.size();
    std::vector<scalar> out(n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
    {
        scalar s = 0;
        for (label k = a.srcOffset[i]; k < a.srcOffset[i+1]; ++k) s += a.weight[k] * psi[a.nbrCell[k]];
        out[i] = s;
    }
    return out;
}


std::vector<vector> interpolateVec(const AMIInterface& a, const std::vector<vector>& U)
{
    const std::size_t n = a.ownCell.size();
    const tensor& T = a.forwardT;
    std::vector<vector> out(n, vector{0, 0, 0});
    for (std::size_t i = 0; i < n; ++i)
    {
        vector s{0, 0, 0};
        for (label k = a.srcOffset[i]; k < a.srcOffset[i+1]; ++k)
        {
            const vector& v = U[a.nbrCell[k]];
            // forwardT applied BEFORE the weighted sum, matching amiInterpVecKernel. The transform is
            // constant per interface so the order does not change the value; it changes whether this is
            // a transcription of the kernel or a paraphrase of it.
            const vector r{T.xx*v.x + T.xy*v.y + T.xz*v.z,
                           T.yx*v.x + T.yy*v.y + T.yz*v.z,
                           T.zx*v.x + T.zy*v.y + T.zz*v.z};
            const scalar w = a.weight[k];
            s.x += w * (a.translational ? v.x : r.x);
            s.y += w * (a.translational ? v.y : r.y);
            s.z += w * (a.translational ? v.z : r.z);
        }
        out[i] = s;
    }
    return out;
}


std::vector<scalar> faceValue(const AMIInterface& a, const std::vector<scalar>& cell)
{
    const std::size_t n = a.ownCell.size();
    const std::vector<scalar> nbr = interpolate(a, cell);
    std::vector<scalar> out(n);
    for (std::size_t i = 0; i < n; ++i)
        out[i] = a.weights[i] * cell[a.ownCell[i]] + (1.0 - a.weights[i]) * nbr[i];
    return out;
}


void assembleMomentum(
    const AMIInterface&        a,
    const std::vector<scalar>& nuEffCell,
    const std::vector<scalar>& phi,
    State&                     st,
    std::vector<scalar>&       diag)
{
    const std::size_t n = a.ownCell.size();
    const std::vector<scalar> nuN = interpolate(a, nuEffCell);
    st.ifCoeff.assign(n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
    {
        const label o = a.ownCell[i];
        // The face diffusivity is the INTERPOLATED one -- w*nuEff[own] + (1-w)*interp(nuEff) -- not the
        // owner cell's, exactly as an internal face takes fvc::interpolate(nuEff). Using the owner value
        // is the same class of error that put 90% of pitzDaily's epsilon residual on one patch.
        const scalar lap = (a.weights[i] * nuEffCell[o] + (1.0 - a.weights[i]) * nuN[i])
                         * a.deltaCoeffs[i] * a.magSf[i];
        const scalar p = phi[i];
        // Upwind convection split: the off-diagonal takes the INFLOW half, the diagonal the outflow half.
        st.ifCoeff[i] = -lap + std::fmin(p, 0.0);
        diag[o]      += lap + std::fmax(p, 0.0);
    }
}


void assembleLaplacian(
    const AMIInterface&        a,
    const std::vector<scalar>& gammaCell,
    State&                     st,
    std::vector<scalar>&       diag,
    const bool                 addToDiag)
{
    const std::size_t n = a.ownCell.size();
    const std::vector<scalar> gN = interpolate(a, gammaCell);
    st.ifCoeff.assign(n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
    {
        const label o = a.ownCell[i];
        const scalar c = (a.weights[i] * gammaCell[o] + (1.0 - a.weights[i]) * gN[i])
                       * a.deltaCoeffs[i] * a.magSf[i];
        // THE SIGN IS THE OPPOSITE OF assembleMomentum's, and that is not an inconsistency: the two
        // equations carry the laplacian with opposite signs. Momentum is
        //     fvm::div(phi,U) - fvm::laplacian(nuEff,U)          -> ifCoeff = -lap, diag += lap
        // while the pressure equation is
        //     fvm::laplacian(rAUf,p) == fvc::div(phiHbyA)        -> ifCoeff = +c,   diag -= c
        // This reference had it backwards on the first writing, copied across from the momentum case,
        // and the stage gate caught it on the first run -- which is the whole argument for having one.
        st.ifCoeff[i] = c;
        if (addToDiag) diag[o] -= c;
    }
}


void amul(
    const AMIInterface&        a,
    const State&               st,
    const std::vector<scalar>& psi,
    std::vector<scalar>&       Apsi)
{
    const std::vector<scalar> pn = interpolate(a, psi);
    for (std::size_t i = 0; i < a.ownCell.size(); ++i)
        Apsi[a.ownCell[i]] += st.ifCoeff[i] * pn[i];
}


void addH(
    const AMIInterface&        a,
    const State&               st,
    const std::vector<scalar>& UN,
    const std::vector<scalar>& V,
    std::vector<scalar>&       H)
{
    for (std::size_t i = 0; i < a.ownCell.size(); ++i)
    {
        const label o = a.ownCell[i];
        H[o] -= st.ifCoeff[i] * UN[i] / V[o];
    }
}


void flux(const AMIInterface& a, const std::vector<vector>& HbyA, State& st)
{
    const std::size_t n = a.ownCell.size();
    const std::vector<vector> Hn = interpolateVec(a, HbyA);
    st.phi.assign(n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
    {
        const label o = a.ownCell[i];
        const scalar w = a.weights[i];
        const vector f{w * HbyA[o].x + (1.0 - w) * Hn[i].x,
                       w * HbyA[o].y + (1.0 - w) * Hn[i].y,
                       w * HbyA[o].z + (1.0 - w) * Hn[i].z};
        st.phi[i] = f.x * a.Sf[i].x + f.y * a.Sf[i].y + f.z * a.Sf[i].z;
    }
}


void correctFlux(
    const AMIInterface&        a,
    const State&               st,
    const std::vector<scalar>& p,
    State&                     phiOut)
{
    const std::vector<scalar> pn = interpolate(a, p);
    phiOut.phi.resize(a.ownCell.size());
    for (std::size_t i = 0; i < a.ownCell.size(); ++i)
        phiOut.phi[i] = st.phi[i] - st.ifCoeff[i] * (pn[i] - p[a.ownCell[i]]);
}


void addDiv(
    const AMIInterface&        a,
    const State&               st,
    const std::vector<scalar>& V,
    std::vector<scalar>&       div)
{
    for (std::size_t i = 0; i < a.ownCell.size(); ++i)
    {
        const label o = a.ownCell[i];
        div[o] += st.phi[i] / V[o];
    }
}

} // namespace cyclicAMI
} // namespace cpu
} // namespace brae
