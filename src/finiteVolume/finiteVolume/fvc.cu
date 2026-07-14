#include "fvc.cuh"

namespace brae {
namespace fvc {

std::vector<vector> gaussGrad(
    const GeometricField<scalar>& p,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches)
{
    const label nC  = m.nCells();
    const label nIf = m.nInternalFaces();
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();
    const std::vector<scalar>& w  = g.weights();
    const std::vector<vector>& Sf = g.Sf();

    std::vector<vector> grad(nC, vector{0, 0, 0});

    // Internal faces: linear interpolation to the face, +owner / -neighbour.
    for (label f = 0; f < nIf; ++f)
    {
        const label o = own[f], n = nei[f];
        const scalar pf = w[f] * p.internal[o] + (1.0 - w[f]) * p.internal[n];
        const vector Sfssf = Sf[f] * pf;
        grad[o] += Sfssf;
        grad[n] = grad[n] - Sfssf;
    }

    // Boundary faces: use the evaluated boundary face value.
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const FvPatch& fp = patches[pi];
        const std::vector<scalar>& pv = p.boundary[pi]->value();
        for (label i = 0; i < fp.size; ++i)
            grad[fp.faceCells[i]] += Sf[fp.start + i] * pv[i];
    }

    for (label c = 0; c < nC; ++c)
        grad[c] = grad[c] / g.V()[c];
    return grad;
}

std::vector<tensor> gaussGrad(
    const GeometricField<vector>& U,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches)
{
    const label nC = m.nCells(), nIf = m.nInternalFaces();
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();
    const std::vector<scalar>& w  = g.weights();
    const std::vector<vector>& Sf = g.Sf();

    std::vector<tensor> grad(nC, tensor{0,0,0,0,0,0,0,0,0});
    for (label f = 0; f < nIf; ++f)
    {
        const vector Uf = w[f] * U.internal[own[f]] + (1.0 - w[f]) * U.internal[nei[f]];
        const tensor SfUf = outer(Sf[f], Uf);
        grad[own[f]] += SfUf;
        grad[nei[f]] = grad[nei[f]] - SfUf;
    }
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const FvPatch& fp = patches[pi];
        const std::vector<vector>& uv = U.boundary[pi]->value();
        for (label i = 0; i < fp.size; ++i)
            grad[fp.faceCells[i]] += outer(Sf[fp.start + i], uv[i]);
    }
    for (label c = 0; c < nC; ++c)
        grad[c] = grad[c] / g.V()[c];
    return grad;
}

SurfaceScalarField flux(
    const GeometricField<vector>& U,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches)
{
    const label nIf = m.nInternalFaces();
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();
    const std::vector<scalar>& w  = g.weights();
    const std::vector<vector>& Sf = g.Sf();

    SurfaceScalarField phi;
    phi.internal.resize(nIf);
    for (label f = 0; f < nIf; ++f)
    {
        const vector Uf = w[f] * U.internal[own[f]] + (1.0 - w[f]) * U.internal[nei[f]];
        phi.internal[f] = dot(Uf, Sf[f]);
    }
    phi.boundary.resize(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const FvPatch& fp = patches[pi];
        const std::vector<vector>& uv = U.boundary[pi]->value();
        phi.boundary[pi].resize(fp.size);
        for (label i = 0; i < fp.size; ++i)
            phi.boundary[pi][i] = dot(uv[i], Sf[fp.start + i]);
    }
    return phi;
}

SurfaceScalarField interpolate(
    const std::vector<scalar>& vol,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches)
{
    const label nIf = m.nInternalFaces();
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();
    const std::vector<scalar>& w  = g.weights();

    SurfaceScalarField sf;
    sf.internal.resize(nIf);
    for (label f = 0; f < nIf; ++f)
        sf.internal[f] = w[f] * vol[own[f]] + (1.0 - w[f]) * vol[nei[f]];
    sf.boundary.resize(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        sf.boundary[pi].resize(patches[pi].size);
        for (label i = 0; i < patches[pi].size; ++i)
            sf.boundary[pi][i] = vol[patches[pi].faceCells[i]];
    }
    return sf;
}

std::vector<scalar> div(
    const SurfaceScalarField& phi,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches)
{
    const label nC = m.nCells(), nIf = m.nInternalFaces();
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();

    std::vector<scalar> d(nC, 0.0);
    for (label f = 0; f < nIf; ++f)
    {
        d[own[f]] += phi.internal[f];
        d[nei[f]] -= phi.internal[f];
    }
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
        for (label i = 0; i < patches[pi].size; ++i)
            d[patches[pi].faceCells[i]] += phi.boundary[pi][i];
    for (label c = 0; c < nC; ++c)
        d[c] /= g.V()[c];
    return d;
}

std::vector<std::vector<tensor>> gradUBoundary(
    const GeometricField<vector>& U,
    const std::vector<tensor>& gradUcell,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches)
{
    const std::vector<vector>& Sf    = g.Sf();
    const std::vector<scalar>& magSf = g.magSf();
    std::vector<std::vector<tensor>> gb(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const FvPatch& fp = patches[pi];
        gb[pi].assign(fp.size, tensor{0,0,0,0,0,0,0,0,0});
        if (fp.type == "empty") continue;   // empty patches contribute nothing to fvc operations
        const std::vector<vector>& uv = U.boundary[pi]->value();
        for (label i = 0; i < fp.size; ++i)
        {
            const label c  = fp.faceCells[i];
            const label gf = fp.start + i;
            const vector n = (1.0 / magSf[gf]) * Sf[gf];                       // unit normal
            const vector snGrad = (uv[i] - U.internal[c]) * fp.deltaCoeffs[i]; // (U_b - U_c)/d
            const tensor& gc = gradUcell[c];                                   // extrapolated cell grad
            gb[pi][i] = gc + outer(n, snGrad - dot(n, gc));                    // normal comp -> snGrad
        }
    }
    return gb;
}

std::vector<vector> div(
    const std::vector<tensor>& tCell,
    const std::vector<std::vector<tensor>>& tBnd,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches)
{
    const label nC = m.nCells(), nIf = m.nInternalFaces();
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();
    const std::vector<scalar>& w  = g.weights();
    const std::vector<vector>& Sf = g.Sf();

    std::vector<vector> d(nC, vector{0.0, 0.0, 0.0});
    for (label f = 0; f < nIf; ++f)
    {
        const tensor Tf = w[f] * tCell[own[f]] + (1.0 - w[f]) * tCell[nei[f]];  // linear interpolate
        const vector SfTf = dot(Sf[f], Tf);                                     // Sf & T_f
        d[own[f]] += SfTf;
        d[nei[f]] = d[nei[f]] - SfTf;
    }
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (patches[pi].type == "empty") continue;   // empty patches excluded (OF fvc semantics)
        for (label i = 0; i < patches[pi].size; ++i)
            d[patches[pi].faceCells[i]] += dot(Sf[patches[pi].start + i], tBnd[pi][i]);
    }
    for (label c = 0; c < nC; ++c)
        d[c] = d[c] / g.V()[c];
    return d;
}

} // namespace fvc
} // namespace brae
