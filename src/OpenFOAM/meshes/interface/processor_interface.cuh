#pragma once
// brae::ProcessorInterface, rank<->rank coupled interface (was an internal face before
// decomposition). Mirrors OpenFOAM processorLduInterface / processorLduInterfaceField.
// CPU backend exchanges via brae::Pstream (MPI); the GPU backend will swap the transport
// for DSM (intra-cluster) / global (inter-cluster) with identical call ordering.
#include "ldu_interface.cuh"
#include "cf_pstream.cuh"
#include <vector>

namespace brae {

class ProcessorInterface : public lduInterface
{
public:
    ProcessorInterface(int myPart, int nbrPart, std::vector<label> faceCells, int tag = 0);

    int  myProcNo()     const { return myPart_; }
    int  neighbProcNo() const { return nbrPart_; }
    label size()        const { return static_cast<label>(faceCells_.size()); }

    const std::vector<label>& faceCells() const override { return faceCells_; }

    void interfaceInternalField(const scalar* psi, std::vector<scalar>& out) const override;
    void initInterfaceMatrixUpdate(const scalar* psi) override;       // posts irecv + isend
    void updateInterfaceMatrix(scalar* result, const scalar* coeffs) override;

    // Neighbour boundary values delivered by the last exchange (valid after Pstream::waitAll).
    const std::vector<scalar>& neighbourField() const { return recvBuf_; }

private:
    int                 myPart_;
    int                 nbrPart_;
    int                 tag_;
    std::vector<label>  faceCells_;
    std::vector<scalar> sendBuf_;
    std::vector<scalar> recvBuf_;
};

} // namespace brae
