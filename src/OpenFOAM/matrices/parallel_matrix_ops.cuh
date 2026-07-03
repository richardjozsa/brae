#pragma once
// cf parallel fvMatrix A()/H() for the SIMPLE p-U coupling on a distributed matrix. Mirrors
// fvMatrix::A()/H() + lduMatrix::H(), with the processor-interface coupling folded into A.psi:
//   A()   = (diag + boundary internalCoeffs) / V           (= L.diagC / V)
//   H(psi)= ( lduMatrix::H(psi) + source + boundarySource ) / V
//   lduMatrix::H(psi) = -offdiag.psi = diag.psi - A.psi     (A.psi via parallelAmul incl. interfaces)
// For a scalar matrix the boundary-diagonal correction term ((cmptAv(ic)-ic)*psi) is zero; the
// momentum (vector) loop applies H per component with the same scalar matrix + per-component source.
#include "parallel_amul.cuh"
#include "ldu_matrix.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fvc.cuh"   // SurfaceScalarField
#include "local_mesh.cuh"
#include <cmath>
#include <vector>

namespace brae {

// pEqn.flux(): conservative face flux of a solved distributed matrix. Internal/real-boundary as
// serial fvMatrix::flux() (the real-boundary coeffs come from the serially-assembled local Mlocal);
// a processor face carries -interfaceCoeff*(p_halo - p_local) (= the cut-face laplacian flux).
inline SurfaceScalarField parallelMatrixFlux(const FvScalarMatrix& Mlocal, const DistributedMatrix& L,
                                             const GeometricField<scalar>& p, const PrimitiveMesh& m,
                                             const std::vector<FvPatch>& lpatches) {
    const label nIf = m.nInternalFaces();
    SurfaceScalarField flux;
    flux.internal.resize(nIf);
    for (label f = 0; f < nIf; ++f)
        flux.internal[f] = L.upper[f] * p.internal[L.upperAddr[f]] - L.lower[f] * p.internal[L.lowerAddr[f]];
    flux.boundary.resize(lpatches.size());
    std::size_t pj = 0;
    for (std::size_t pi = 0; pi < lpatches.size(); ++pi) {
        flux.boundary[pi].resize(lpatches[pi].size);
        if (lpatches[pi].type == "processor") {
            const std::vector<scalar>& halo = p.boundary[pi]->patchNeighbourField();
            for (label i = 0; i < lpatches[pi].size; ++i)
                flux.boundary[pi][i] = -L.interfaceCoeffs[pj][i] * (halo[i] - p.internal[lpatches[pi].faceCells[i]]);
            ++pj;
        } else {
            for (label i = 0; i < lpatches[pi].size; ++i)
                flux.boundary[pi][i] = Mlocal.internalCoeffs[pi][i] * p.internal[lpatches[pi].faceCells[i]] - Mlocal.boundaryCoeffs[pi][i];
        }
    }
    return flux;
}

// fvMatrix::relax(alpha) for a distributed matrix: diagonal-dominance + under-relaxation. Same as
// serial relaxMatrix but sumOff (the |off-diagonal| row sum) ALSO includes the processor-interface
// coeffs. Modifies L.diag (the matrix diagonal) and Mlocal.source (the explicit source). The local
// FvMatrix carries the real-boundary internalCoeffs (processor patches contribute zero).
template <typename T>
void parallelRelaxMatrix(DistributedMatrix& L, FvMatrix<T>& Mlocal, const std::vector<T>& psi,
                         const LocalMesh& lm, const std::vector<FvPatch>& lpatches, scalar alpha) {
    if (alpha <= 0.0) return;
    const label nC = lm.mesh.nCells(), nIf = lm.mesh.nInternalFaces();
    const std::vector<scalar> D0 = L.diag;
    std::vector<scalar> sumOff(nC, 0.0);
    for (label f = 0; f < nIf; ++f) { sumOff[L.lowerAddr[f]] += std::fabs(L.upper[f]); sumOff[L.upperAddr[f]] += std::fabs(L.lower[f]); }
    std::size_t pj = 0;
    for (const FvPatch& p : lpatches) {
        if (p.type != "processor") continue;
        for (label i = 0; i < p.size; ++i) sumOff[p.faceCells[i]] += std::fabs(L.interfaceCoeffs[pj][i]);
        ++pj;
    }
    for (std::size_t pi = 0; pi < lpatches.size(); ++pi)
        for (label i = 0; i < lpatches[pi].size; ++i)
            L.diag[lpatches[pi].faceCells[i]] += cmptMagMax(Mlocal.internalCoeffs[pi][i]);
    for (label c = 0; c < nC; ++c) L.diag[c] = std::fmax(std::fabs(L.diag[c]), sumOff[c]) / alpha;
    for (std::size_t pi = 0; pi < lpatches.size(); ++pi)
        for (label i = 0; i < lpatches[pi].size; ++i)
            L.diag[lpatches[pi].faceCells[i]] -= cmptMin(Mlocal.internalCoeffs[pi][i]);
    for (label c = 0; c < nC; ++c) Mlocal.source[c] += (L.diag[c] - D0[c]) * psi[c];
}

inline std::vector<scalar> parallelMatrixA(const DistributedMatrix& L, const std::vector<scalar>& V) {
    std::vector<scalar> A(L.diagC.size());
    for (std::size_t c = 0; c < A.size(); ++c) A[c] = L.diagC[c] / V[c];
    return A;
}

inline std::vector<scalar> parallelMatrixH(const DistributedMatrix& L, const std::vector<scalar>& psi,
                                           std::vector<ProcessorInterface>& interfaces,
                                           const std::vector<scalar>& V) {
    const std::vector<scalar> Apsi = parallelAmul(L, psi, interfaces);   // diag.psi + offdiag (incl. interface)
    std::vector<scalar> H(psi.size());
    for (std::size_t c = 0; c < psi.size(); ++c) H[c] = (L.diag[c] * psi[c] - Apsi[c] + L.b[c]) / V[c];
    return H;
}

} // namespace brae
