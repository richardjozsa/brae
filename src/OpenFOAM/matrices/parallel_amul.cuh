#pragma once
// cf parallel matrix-vector product. Mirrors OpenFOAM lduMatrix::Amul: the internal (local)
// product diag*psi + upper/lower contributions, bracketed by the coupled-interface master loop
//   initMatrixInterfaces (post all exchanges)  ->  internal product  ->  updateMatrixInterfaces
//     (wait, then result[faceCell] -= bouCoeff * psiNeighbour for each interface).
// A processor cut face is the SAME arithmetic as an internal face; the neighbour psi is exchanged
// rather than read locally. The interface boundary coefficient is the negated off-diagonal of the
// cut face (-upper if the local cell is the owner, -lower if it is the neighbour), so that
// "result -= bouCoeff*psiNbr" reproduces the internal-face "+upper*psiNbr".
#include "cf_types.cuh"
#include "ldu_matrix.cuh"
#include "domain_decomposition.cuh"
#include "processor_interface.cuh"
#include "cf_pstream.cuh"
#include <vector>

namespace brae {

// The local (per-partition) view of a distributed lduMatrix: local diag + local internal-face
// upper/lower (local cell addressing) + one boundary-coeff list per processor interface.
struct DistributedMatrix
{
    std::vector<scalar>              diag;
    std::vector<label>               lowerAddr, upperAddr;   // = decomp.localOwner/localNeighbour
    std::vector<scalar>              upper, lower;
    std::vector<std::vector<scalar>> interfaceCoeffs;        // per interface (decomp.interfaces() order)
    std::vector<scalar>              diagC, b;               // boundary folded in (addBoundaryDiag/Source)
};

// Build the local matrix for `decomp`'s partition from the GLOBAL matrix (validation/bring-up:
// the global matrix is assembled serially, then sliced by index, same numbers a local assembly
// would produce). The interface coeff = -upper (owner side) / -lower (neighbour side).
inline DistributedMatrix distribute(const FvScalarMatrix& global, const DomainDecomposition& decomp)
{
    DistributedMatrix L;
    const std::vector<label>& addr = decomp.cellProcAddressing();
    L.diag.resize(addr.size());
    for (std::size_t lc = 0; lc < addr.size(); ++lc) L.diag[lc] = global.diag[addr[lc]];

    L.lowerAddr = decomp.localOwner();
    L.upperAddr = decomp.localNeighbour();
    const std::vector<label>& lfg = decomp.localFaceGlobal();
    L.upper.resize(lfg.size());
    L.lower.resize(lfg.size());
    for (std::size_t f = 0; f < lfg.size(); ++f)
    {
        L.upper[f] = global.upper[lfg[f]];
        L.lower[f] = global.lower[lfg[f]];
    }

    const auto& ifg  = decomp.interfaceGlobalFaces();
    const auto& side = decomp.interfaceOwnerSide();
    L.interfaceCoeffs.resize(ifg.size());
    for (std::size_t i = 0; i < ifg.size(); ++i)
    {
        L.interfaceCoeffs[i].resize(ifg[i].size());
        for (std::size_t f = 0; f < ifg[i].size(); ++f)
        {
            const label gf = ifg[i][f];
            L.interfaceCoeffs[i][f] = -(side[i][f] ? global.upper[gf] : global.lower[gf]);
        }
    }
    return L;
}

// Fold the REAL boundary patches (inlet/outlet/walls, NOT processor interfaces, which the
// decomposition created and parallelAmul handles) into the local diagonal + source, and slice the
// global source. Mirrors fvMatrix addBoundaryDiag/addBoundarySource restricted to this partition.
inline void foldBoundary(
    DistributedMatrix& L,
    const FvScalarMatrix& global,
    const std::vector<FvPatch>& gPatches,
    const std::vector<label>& addr,
    const std::vector<label>& cellToPart,
    int myPart)
{
    std::vector<label> g2l(global.diag.size(), -1);
    for (std::size_t lc = 0; lc < addr.size(); ++lc) g2l[addr[lc]] = static_cast<label>(lc);

    L.diagC = L.diag;
    L.b.resize(addr.size());
    for (std::size_t lc = 0; lc < addr.size(); ++lc) L.b[lc] = global.source[addr[lc]];
    for (std::size_t pi = 0; pi < gPatches.size(); ++pi)
        for (label i = 0; i < gPatches[pi].size; ++i)
        {
            const label gc = gPatches[pi].faceCells[i];
            if (cellToPart[gc] == myPart)
            {
                const label lc = g2l[gc];
                L.diagC[lc] += global.internalCoeffs[pi][i];
                L.b[lc]     += global.boundaryCoeffs[pi][i];
            }
        }
}

// L += s*R for two distributed matrices over the SAME local addressing (e.g. convection +
// (-1)*laplacian for the momentum). diag/upper/lower/interfaceCoeffs all combine termwise.
inline void addEqualDist(DistributedMatrix& L, const DistributedMatrix& R, scalar s)
{
    for (std::size_t c = 0; c < L.diag.size(); ++c)  L.diag[c]  += s * R.diag[c];
    for (std::size_t f = 0; f < L.upper.size(); ++f)
    {
        L.upper[f] += s * R.upper[f];
        L.lower[f] += s * R.lower[f];
    }
    for (std::size_t j = 0; j < L.interfaceCoeffs.size(); ++j)
        for (std::size_t i = 0; i < L.interfaceCoeffs[j].size(); ++i)
            L.interfaceCoeffs[j][i] += s * R.interfaceCoeffs[j][i];
}

// Apsi = A.psi for the distributed matrix, with the processor-interface coupling (one MPI exchange).
inline std::vector<scalar> parallelAmul(
    const DistributedMatrix& L,
    const std::vector<scalar>& psi,
    std::vector<ProcessorInterface>& interfaces)
{
    for (auto& itf : interfaces) itf.initInterfaceMatrixUpdate(psi.data());   // post all exchanges

    std::vector<scalar> Apsi(L.diag.size());
    for (std::size_t c = 0; c < L.diag.size(); ++c) Apsi[c] = L.diag[c] * psi[c];
    for (std::size_t f = 0; f < L.upperAddr.size(); ++f)
    {
        Apsi[L.upperAddr[f]] += L.lower[f] * psi[L.lowerAddr[f]];
        Apsi[L.lowerAddr[f]] += L.upper[f] * psi[L.upperAddr[f]];
    }

    Pstream::waitAll();                                                       // complete exchanges
    for (std::size_t i = 0; i < interfaces.size(); ++i)
        interfaces[i].updateInterfaceMatrix(Apsi.data(), L.interfaceCoeffs[i].data());
    return Apsi;
}

} // namespace brae
