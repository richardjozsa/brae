#include "ldu_spmv.cuh"

namespace brae {

std::vector<scalar> Amul(
    const FvScalarMatrix& M,
    const std::vector<scalar>& psi,
    const PrimitiveMesh& m)
{
    const label nC  = m.nCells();
    const label nIf = m.nInternalFaces();
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();

    std::vector<scalar> Apsi(nC);
    for (label c = 0; c < nC; ++c) Apsi[c] = M.diag[c] * psi[c];

    for (label f = 0; f < nIf; ++f)
    {
        Apsi[nei[f]] += M.lower[f] * psi[own[f]];
        Apsi[own[f]] += M.upper[f] * psi[nei[f]];
    }
    return Apsi;
}

} // namespace brae
