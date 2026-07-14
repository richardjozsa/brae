#pragma once
// brae::lduInterface, abstract coupled interface. Mirrors OpenFOAM
// lduInterface / lduInterfaceField (src/OpenFOAM/.../lduAddressing/lduInterface).
//
// The unifying contract (the whole port in three calls): a coupled-interface face is the
// SAME arithmetic as an internal face; only the neighbour value's source differs (local
// array vs exchanged buffer). Concrete kinds: ProcessorInterface (rank<->rank, MPI now /
// DSM later) and, later, CyclicInterface (periodic).
#include "cf_types.cuh"
#include <vector>

namespace brae {

class lduInterface
{
public:
    virtual ~lduInterface() = default;

    // Local cells adjacent to the interface faces (the gather index; the lowerAddr role).
    virtual const std::vector<label>& faceCells() const = 0;

    // Gather psi at faceCells into the interface's outgoing buffer.
    virtual void interfaceInternalField(const scalar* psi, std::vector<scalar>& out) const = 0;

    // Post the exchange (transport-specific: MPI Isend/Irecv now, DSM write later).
    virtual void initInterfaceMatrixUpdate(const scalar* psi) = 0;

    // Complete the exchange and apply the coupling:
    //   result[faceCells[f]] -= coeffs[f] * psiNeighbour[f]   (OpenFOAM A.psi sign convention)
    virtual void updateInterfaceMatrix(scalar* result, const scalar* coeffs) = 0;
};

} // namespace brae
