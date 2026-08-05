// lduInterface increment 4d: parallel fvMatrix A()/H() (the SIMPLE p-U coupling operations) on a
// distributed matrix reproduce the serial A()/H(). H carries the processor-interface coupling for
// free via parallelAmul. Run: mpirun -np {1,2,4,8} test_parallel_matrixH <caseDir>
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "fvm.cuh"
#include "fvc.cuh"
#include "fv_matrix_ops.cuh"
#include "scotch_decomposition.cuh"
#include "local_mesh.cuh"
#include "local_assembly.cuh"
#include "parallel_amul.cuh"
#include "parallel_matrix_ops.cuh"
#include "cf_pstream.cuh"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace brae;

// Serial scalar fvMatrix::H(psi) (no boundary-diag term for scalars).
static std::vector<scalar> serialScalarH(const FvScalarMatrix& M, const std::vector<scalar>& psi,
                                         const PrimitiveMesh& m, const FvGeometry& g, const std::vector<FvPatch>& patches) {
    const label nC = m.nCells(), nIf = m.nInternalFaces();
    const std::vector<label>& own = m.owner(); const std::vector<label>& nei = m.neighbour();
    std::vector<scalar> H(nC, 0.0);
    for (label f = 0; f < nIf; ++f) { H[nei[f]] -= M.lower[f] * psi[own[f]]; H[own[f]] -= M.upper[f] * psi[nei[f]]; }
    for (label c = 0; c < nC; ++c) H[c] += M.source[c];
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
        for (label i = 0; i < patches[pi].size; ++i) H[patches[pi].faceCells[i]] += M.boundaryCoeffs[pi][i];
    for (label c = 0; c < nC; ++c) H[c] /= g.V()[c];
    return H;
}

int main(int argc, char** argv) {
    Pstream::init(argc, argv);
    const int rank = Pstream::myProcNo(), nproc = Pstream::nProcs();
    if (argc < 2) { if (Pstream::master()) std::printf("usage: %s <caseDir>\n", argv[0]); Pstream::finalize(); return 2; }
    const std::string caseDir = argv[1];

    PrimitiveMesh gm;  gm.read(caseDir + "/constant/polyMesh");
    FvGeometry gg;     gg.build(gm);
    const std::vector<FvPatch> gpatches = buildPatches(gm, gg);
    const label nC = gm.nCells(), nIf = gm.nInternalFaces();

    GeometricField<vector> U = buildField<vector>(readField<vector>(caseDir + "/282/U"), gpatches, nC);  U.evaluateBoundary();
    GeometricField<scalar> T = buildField<scalar>(readField<scalar>(caseDir + "/282/p"), gpatches, nC);  T.evaluateBoundary();
    const SurfaceScalarField phi = fvc::flux(U, gm, gg, gpatches);
    FvScalarMatrix M = fvm::div(phi.internal, phi.boundary, T, gm, gpatches);
    addEqual(M, fvm::laplacian(T, 1.0, gm, gg, gpatches), -1.0);
    for (label c = 0; c < nC; ++c) M.source[c] = gg.V()[c] * (1.0 + 0.5 * std::sin(40.0 * gg.C()[c].x));

    std::vector<scalar> psi(nC);
    for (label c = 0; c < nC; ++c) psi[c] = std::sin(25.0 * gg.C()[c].x) + 0.3 * gg.C()[c].y;
    const std::vector<scalar> Aserial = matrixA(M, gm, gg, gpatches);
    const std::vector<scalar> Hserial = serialScalarH(M, psi, gm, gg, gpatches);

    std::vector<label> cellToPart(nC, 0);
    if (Pstream::master()) cellToPart = scotchDecompose(gm, nproc);
    Pstream::broadcast(cellToPart.data(), nC, 0);
    LocalMesh Lm = buildLocalMesh(gm, cellToPart, rank);
    FvGeometry lg; lg.build(Lm.mesh);
    const std::vector<FvPatch> lpatches = buildPatches(Lm.mesh, lg);
    const auto procDelta = computeProcDeltaCoeffs(Lm, lg, lpatches);
    std::vector<scalar> localPhi(Lm.mesh.nFaces(), 0.0), localGammaF(Lm.mesh.nFaces(), 0.0);
    for (label f = 0; f < Lm.mesh.nFaces(); ++f) { const label gf = Lm.faceGlobal[f]; if (gf < nIf) { localPhi[f] = phi.internal[gf]*(Lm.faceFlip[f]?-1.0:1.0); localGammaF[f] = 1.0; } }
    DistributedMatrix L = assembleLocalConvection(Lm, lpatches, localPhi);
    addEqualDist(L, assembleLocalLaplacianF(Lm, lg, lpatches, procDelta, localGammaF), -1.0);
    foldBoundary(L, M, gpatches, Lm.cellProcAddr, cellToPart, rank);

    std::vector<ProcessorInterface> interfaces;
    for (std::size_t j = 0; j < Lm.procNbr.size(); ++j) interfaces.emplace_back(rank, Lm.procNbr[j], Lm.procFaceCells[j]);
    std::vector<scalar> psiL(Lm.mesh.nCells());
    for (label lc = 0; lc < Lm.mesh.nCells(); ++lc) psiL[lc] = psi[Lm.cellProcAddr[lc]];

    const std::vector<scalar> Alocal = parallelMatrixA(L, lg.V());
    const std::vector<scalar> Hlocal = parallelMatrixH(L, psiL, interfaces, lg.V());

    scalar mA = 0, mH = 0, gA = 0, gH = 0;
    for (label lc = 0; lc < Lm.mesh.nCells(); ++lc) {
        const label gc = Lm.cellProcAddr[lc];
        mA = std::fmax(mA, std::fabs(Alocal[lc] - Aserial[gc])); gA = std::fmax(gA, std::fabs(Aserial[gc]));
        mH = std::fmax(mH, std::fabs(Hlocal[lc] - Hserial[gc])); gH = std::fmax(gH, std::fabs(Hserial[gc]));
    }
    const scalar relA = Pstream::allReduce(mA, ReduceOp::Max) / Pstream::allReduce(gA, ReduceOp::Max);
    const scalar relH = Pstream::allReduce(mH, ReduceOp::Max) / Pstream::allReduce(gH, ReduceOp::Max);

    if (Pstream::master()) {
        std::printf("test_parallel_matrixH np=%d: A() rel=%.3e  H() rel=%.3e\n", nproc, relA, relH);
        std::printf("%s\n", (relA < 1e-12 && relH < 1e-12) ? "PASS" : "FAIL");
    }
    Pstream::finalize();
    return 0;
}
