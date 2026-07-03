#pragma once
// cf local parallel matrix assembly support. A processor (cut) face is geometrically a boundary
// face of the local mesh, but in the matrix it couples the local cell to the REMOTE cell across the
// interface, so its delta coefficient must be cell-to-cell. cf's internal-face deltaCoeffs = 1/|C_nei
// - C_own| (full distance), so we exchange the REMOTE cell centre per face and form 1/|C_remote -
// C_local|. The cut-face cell centres are in the same global frame (no transform for processor
// interfaces). Matched face ordering -> face i <-> face i. Mirrors OpenFOAM processor delta exchange.
#include "cf_types.cuh"
#include "fv_patch.cuh"
#include "fv_geometry.cuh"
#include "ldu_matrix.cuh"
#include "local_mesh.cuh"
#include "parallel_amul.cuh"
#include "cf_pstream.cuh"
#include <vector>

namespace brae {

// procDeltaCoeffs[j][i] = cell-to-cell delta coeff of processor patch j, face i (j in the order the
// processor patches appear in lpatches, == LocalMesh.procNbr order).
inline std::vector<std::vector<scalar>> computeProcDeltaCoeffs(const LocalMesh& lm, const FvGeometry& lg,
                                                               const std::vector<FvPatch>& lpatches) {
    std::vector<const FvPatch*> pp;
    for (const FvPatch& p : lpatches) if (p.type == "processor") pp.push_back(&p);

    const std::size_t n = pp.size();
    std::vector<std::vector<scalar>> sendC(n), recvC(n);   // 3 scalars (cell centre) per face
    for (std::size_t j = 0; j < n; ++j) {
        const label sz = pp[j]->size;
        sendC[j].resize(3 * sz); recvC[j].resize(3 * sz);
        for (label i = 0; i < sz; ++i) {
            const vector& C = lg.C()[pp[j]->faceCells[i]];
            sendC[j][3 * i] = C.x; sendC[j][3 * i + 1] = C.y; sendC[j][3 * i + 2] = C.z;
        }
        Pstream::irecv(recvC[j].data(), static_cast<int>(3 * sz), lm.procNbr[j], 0);
        Pstream::isend(sendC[j].data(), static_cast<int>(3 * sz), lm.procNbr[j], 0);
    }
    Pstream::waitAll();

    std::vector<std::vector<scalar>> pdc(n);
    for (std::size_t j = 0; j < n; ++j) {
        const label sz = pp[j]->size;
        pdc[j].resize(sz);
        for (label i = 0; i < sz; ++i) {
            const vector Cremote{recvC[j][3 * i], recvC[j][3 * i + 1], recvC[j][3 * i + 2]};
            pdc[j][i] = 1.0 / mag(Cremote - lg.C()[pp[j]->faceCells[i]]);
        }
    }
    return pdc;
}

// Build the scalar DistributedMatrix structure (diag/upper/lower/interfaceCoeffs) for the momentum
// div(phi,U) - laplacian(nuEff,U) from the serially-assembled LOCAL FvMatrix (correct internal +
// real-boundary coeffs; ZERO at processor patches since ProcessorFvPatchField returns zero coeffs)
// plus the processor faces (convection w*phi + (-laplacian) coeff). Mirrors the global cut-face
// off-diagonal: diag += w*phi + coeff; interfaceCoeff = -(1-w)*phi + coeff, coeff=nuEff_f*magSf*procDelta.
template <typename T>
inline DistributedMatrix momentumDistributed(const FvMatrix<T>& Mlocal, const LocalMesh& lm, const FvGeometry& lg,
                                             const std::vector<FvPatch>& lpatches,
                                             const std::vector<scalar>& localPhi, const std::vector<scalar>& localNuEffF,
                                             const std::vector<std::vector<scalar>>& procDelta) {
    const PrimitiveMesh& m = lm.mesh;
    const label nIf = m.nInternalFaces();
    DistributedMatrix L;
    L.diag = Mlocal.diag;
    L.lowerAddr.assign(m.owner().begin(), m.owner().begin() + nIf);
    L.upperAddr = m.neighbour();
    L.upper = Mlocal.upper; L.lower = Mlocal.lower;
    std::size_t pj = 0;
    for (const FvPatch& p : lpatches) {
        if (p.type != "processor") continue;
        std::vector<scalar> ic(p.size);
        for (label i = 0; i < p.size; ++i) {
            const scalar phi = localPhi[p.start + i];
            const scalar w = (phi >= 0.0) ? 1.0 : 0.0;                                   // UPWIND weight
            const scalar coeff = localNuEffF[p.start + i] * lg.magSf()[p.start + i] * procDelta[pj][i];
            L.diag[p.faceCells[i]] += w * phi + coeff;        // convection (outflow) + (-laplacian)
            ic[i] = -(1.0 - w) * phi + coeff;                 // interface: convection inflow + diffusion
        }
        L.interfaceCoeffs.push_back(std::move(ic));
        ++pj;
    }
    return L;
}

// procWeights[j][i] = the linear interpolation weight of the LOCAL cell at processor patch j, face
// i: face value = w*localCell + (1-w)*remoteCell. w = SfdNei/(SfdOwn+SfdNei) with the remote cell
// centre exchanged (matches OpenFOAM surfaceInterpolation weight; reversed cut faces give 1-w so the
// interpolated value equals the global cut-face value either way).
inline std::vector<std::vector<scalar>> computeProcWeights(const LocalMesh& lm, const FvGeometry& lg,
                                                           const std::vector<FvPatch>& lpatches) {
    std::vector<const FvPatch*> pp;
    for (const FvPatch& p : lpatches) if (p.type == "processor") pp.push_back(&p);

    const std::size_t n = pp.size();
    std::vector<std::vector<scalar>> sendC(n), recvC(n);
    for (std::size_t j = 0; j < n; ++j) {
        const label sz = pp[j]->size;
        sendC[j].resize(3 * sz); recvC[j].resize(3 * sz);
        for (label i = 0; i < sz; ++i) {
            const vector& C = lg.C()[pp[j]->faceCells[i]];
            sendC[j][3 * i] = C.x; sendC[j][3 * i + 1] = C.y; sendC[j][3 * i + 2] = C.z;
        }
        Pstream::irecv(recvC[j].data(), static_cast<int>(3 * sz), lm.procNbr[j], 1);
        Pstream::isend(sendC[j].data(), static_cast<int>(3 * sz), lm.procNbr[j], 1);
    }
    Pstream::waitAll();

    std::vector<std::vector<scalar>> w(n);
    label pj = 0;
    for (std::size_t k = 0; k < lpatches.size(); ++k) {
        if (lpatches[k].type != "processor") continue;
        const FvPatch& p = lpatches[k];
        w[pj].resize(p.size);
        for (label i = 0; i < p.size; ++i) {
            const label gf = p.start + i;
            const vector Cremote{recvC[pj][3 * i], recvC[pj][3 * i + 1], recvC[pj][3 * i + 2]};
            const scalar SfdOwn = std::fabs(dot(lg.Sf()[gf], lg.Cf()[gf] - lg.C()[p.faceCells[i]]));
            const scalar SfdNei = std::fabs(dot(lg.Sf()[gf], Cremote - lg.Cf()[gf]));
            w[pj][i] = (SfdOwn + SfdNei > 1e-300) ? SfdNei / (SfdOwn + SfdNei) : 0.5;
        }
        ++pj;
    }
    return w;
}

// Assemble a local orthogonal laplacian(gamma, .) directly from this partition's mesh, as a
// DistributedMatrix: internal faces upper=lower=gamma*magSf*deltaCoeffs (diag -=); processor faces
// become interface coeffs -gamma*magSf*procDelta (diag -= the magnitude), exactly the cut-face
// off-diagonal of the global matrix. Raw matrix (no boundary fold) -> parallelAmul reproduces the
// serial Amul. (Constant gamma; variable gamma would interpolate an exchanged remote gamma.)
// Per-face diffusivity gammaF (size local nFaces). For a processor face, gammaF must be the face-
// interpolated value (interpolate(gammaCell) needs the remote cell gamma -> exchanged like phi).
inline DistributedMatrix assembleLocalLaplacianF(const LocalMesh& lm, const FvGeometry& lg,
                                                 const std::vector<FvPatch>& lpatches,
                                                 const std::vector<std::vector<scalar>>& procDelta,
                                                 const std::vector<scalar>& gammaF) {
    const PrimitiveMesh& m = lm.mesh;
    const label nC = m.nCells(), nIf = m.nInternalFaces();
    const std::vector<scalar>& magSf = lg.magSf();
    const std::vector<scalar>& dc    = lg.deltaCoeffs();

    DistributedMatrix L;
    L.diag.assign(nC, 0.0);
    L.lowerAddr.assign(m.owner().begin(), m.owner().begin() + nIf);
    L.upperAddr = m.neighbour();
    L.upper.resize(nIf); L.lower.resize(nIf);
    for (label f = 0; f < nIf; ++f) {
        const scalar coeff = gammaF[f] * magSf[f] * dc[f];
        L.upper[f] = coeff; L.lower[f] = coeff;
        L.diag[m.owner()[f]]     -= coeff;
        L.diag[m.neighbour()[f]] -= coeff;
    }
    std::size_t pj = 0;
    for (const FvPatch& p : lpatches) {
        if (p.type != "processor") continue;
        std::vector<scalar> ic(p.size);
        for (label i = 0; i < p.size; ++i) {
            const scalar coeff = gammaF[p.start + i] * magSf[p.start + i] * procDelta[pj][i];
            L.diag[p.faceCells[i]] -= coeff;
            ic[i] = -coeff;
        }
        L.interfaceCoeffs.push_back(std::move(ic));
        ++pj;
    }
    return L;
}

// Constant-gamma convenience.
inline DistributedMatrix assembleLocalLaplacian(const LocalMesh& lm, const FvGeometry& lg,
                                                const std::vector<FvPatch>& lpatches,
                                                const std::vector<std::vector<scalar>>& procDelta,
                                                scalar gamma) {
    return assembleLocalLaplacianF(lm, lg, lpatches, procDelta, std::vector<scalar>(lm.mesh.nFaces(), gamma));
}

// Assemble a local upwind convection div(phi,.) directly from this partition's mesh, as a
// DistributedMatrix. localPhi[f] is the flux through local face f consistent with its local outward
// normal (so a processor face uses the OUTWARD flux from the local cell, encoding upwind direction
// uniformly): w=pos0(phi), lower=-w*phi, upper=lower+phi, diag-=lower/upper. A processor face is an
// owner-side face of the local cell: diag += w*phi, interfaceCoeff = -(1-w)*phi (= global cut-face
// -upper/-lower). Raw matrix (real-boundary coeffs folded separately).
inline DistributedMatrix assembleLocalConvection(const LocalMesh& lm, const std::vector<FvPatch>& lpatches,
                                                 const std::vector<scalar>& localPhi) {
    const PrimitiveMesh& m = lm.mesh;
    const label nC = m.nCells(), nIf = m.nInternalFaces();

    DistributedMatrix L;
    L.diag.assign(nC, 0.0);
    L.lowerAddr.assign(m.owner().begin(), m.owner().begin() + nIf);
    L.upperAddr = m.neighbour();
    L.upper.resize(nIf); L.lower.resize(nIf);
    for (label f = 0; f < nIf; ++f) {
        const scalar phi = localPhi[f];
        const scalar w = (phi >= 0.0) ? 1.0 : 0.0;
        L.lower[f] = -w * phi;
        L.upper[f] = L.lower[f] + phi;
        L.diag[m.owner()[f]]     -= L.lower[f];
        L.diag[m.neighbour()[f]] -= L.upper[f];
    }
    for (const FvPatch& p : lpatches) {
        if (p.type != "processor") continue;
        std::vector<scalar> ic(p.size);
        for (label i = 0; i < p.size; ++i) {
            const scalar phi = localPhi[p.start + i];     // outward flux from the local cell
            const scalar w = (phi >= 0.0) ? 1.0 : 0.0;
            L.diag[p.faceCells[i]] += w * phi;            // outflow contributes to the diagonal
            ic[i] = -(1.0 - w) * phi;                     // interface coupling (= -upper)
        }
        L.interfaceCoeffs.push_back(std::move(ic));
    }
    return L;
}

} // namespace brae
