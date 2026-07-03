// Phase 2 gate, processorFvPatchField across the Phase-1 MPI interfaces.
// Decompose pitzDaily; build scalar and vector GeometricFields whose processor patches
// couple the partitions. After evaluateBoundary(), each processor patch's value() must hold
// the NEIGHBOUR cells' values, i.e. the exchanged halo reproduces the serial field.
// psi = global cell index; V = (gid, -gid, 2*gid).
// Run: mpirun -np {2,4} test_processor_field <constant/polyMesh>
#include "primitive_mesh.cuh"
#include "scotch_decomposition.cuh"
#include "domain_decomposition.cuh"
#include "fv_patch.cuh"
#include "fv_patch_field.cuh"
#include "geometric_field.cuh"
#include "cf_pstream.cuh"

#include <cstdio>
#include <memory>
#include <vector>

using namespace brae;

int main(int argc, char** argv) {
    Pstream::init(argc, argv);
    const int rank = Pstream::myProcNo(), nproc = Pstream::nProcs();
    if (argc < 2) { if (Pstream::master()) std::printf("usage\n"); Pstream::finalize(); return 2; }

    PrimitiveMesh m;
    m.read(argv[1]);
    std::vector<label> c2p(m.nCells(), 0);
    if (Pstream::master()) c2p = scotchDecompose(m, nproc);
    Pstream::broadcast(c2p.data(), m.nCells(), 0);
    DomainDecomposition dd(m, c2p, rank);

    GeometricField<scalar> sf;
    GeometricField<vector> vf;
    sf.internal.resize(dd.nLocalCells());
    vf.internal.resize(dd.nLocalCells());
    for (label i = 0; i < dd.nLocalCells(); ++i) {
        const scalar g = static_cast<scalar>(dd.cellProcAddressing()[i]);
        sf.internal[i] = g;
        vf.internal[i] = vector{g, -g, 2 * g};
    }

    // FvPatch per interface (must outlive the patch fields, hence reserve + no realloc).
    std::vector<FvPatch> patches;
    patches.reserve(dd.interfaces().size());
    for (auto& it : dd.interfaces()) {
        FvPatch p;
        p.name = "procBoundary"; p.type = "processor"; p.start = 0; p.size = it.size();
        p.faceCells = it.faceCells();
        patches.push_back(std::move(p));
    }
    for (std::size_t k = 0; k < patches.size(); ++k) {
        const int nbr = dd.interfaces()[k].neighbProcNo();
        sf.boundary.push_back(std::make_unique<ProcessorFvPatchField<scalar>>(patches[k], nbr));
        vf.boundary.push_back(std::make_unique<ProcessorFvPatchField<vector>>(patches[k], nbr));
    }

    sf.evaluateBoundary();
    vf.evaluateBoundary();

    int fails = 0;
    for (std::size_t k = 0; k < patches.size(); ++k) {
        const auto& sv  = sf.boundary[k]->value();
        const auto& vv  = vf.boundary[k]->value();
        const auto& exp = dd.nbrGlobalCells()[k];
        for (std::size_t i = 0; i < exp.size(); ++i) {
            const scalar g = static_cast<scalar>(exp[i]);
            if (sv[i] != g) { if (fails < 5) std::printf("[%d] scalar FAIL got %.0f exp %.0f\n", rank, sv[i], g); ++fails; }
            if (vv[i].x != g || vv[i].y != -g || vv[i].z != 2 * g) {
                if (fails < 5) std::printf("[%d] vector FAIL\n", rank); ++fails;
            }
        }
    }

    const label tf = Pstream::allReduce(static_cast<label>(fails), ReduceOp::Sum);
    if (Pstream::master())
        std::printf("test_processor_field: nproc=%d  %s (fails=%d)\n",
                    nproc, tf == 0 ? "PASS" : "FAIL", (int)tf);
    Pstream::finalize();
    return tf == 0 ? 0 : 1;
}
