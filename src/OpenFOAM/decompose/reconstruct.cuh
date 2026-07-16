#pragma once
// brae reconstructField, decomposePar's inverse: gather a decomposed cell field back to the global
// numbering. Each rank scatters its local values into a global-sized buffer at their cellProcAddr
// positions; a Sum all-reduce assembles the full field on every rank. Each global cell belongs to exactly
// one partition, so the sum double-counts nothing.
//
// This materialises a global-sized host buffer, so it is for OUTPUT / validation, not the solve (the solve
// keeps only per-partition device fields). A master-only point-to-point gather can replace it if the global
// field itself does not fit; that is an output-path optimisation, not a correctness change.
#include "cf_types.cuh"
#include "cf_pstream.cuh"
#include <vector>

namespace brae {

inline std::vector<scalar> reconstructField(
    const std::vector<label>&  cellProcAddr,   // local cell -> global cell
    const std::vector<scalar>& local,          // this rank's field (size == nLocalCells)
    label                      nGlobalCells)
{
    std::vector<scalar> global(nGlobalCells, 0.0);
    for (std::size_t i = 0; i < cellProcAddr.size(); ++i)
        global[cellProcAddr[i]] = local[i];
    Pstream::allReduce(global.data(), static_cast<int>(nGlobalCells), ReduceOp::Sum);
    return global;
}

} // namespace brae
