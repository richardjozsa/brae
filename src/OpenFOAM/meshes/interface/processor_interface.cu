// brae::ProcessorInterface, MPI-backed halo exchange. Re-expressed from OpenFOAM
// processorLduInterfaceField::initInterfaceMatrixUpdate / updateInterfaceMatrix.
#include "processor_interface.cuh"

namespace brae {

ProcessorInterface::ProcessorInterface(int myPart, int nbrPart, std::vector<label> faceCells, int tag)
    : myPart_(myPart),
      nbrPart_(nbrPart),
      tag_(tag),
      faceCells_(std::move(faceCells))
{
    sendBuf_.resize(faceCells_.size());
    recvBuf_.resize(faceCells_.size());
}

void ProcessorInterface::interfaceInternalField(const scalar* psi, std::vector<scalar>& out) const
{
    out.resize(faceCells_.size());
    for (std::size_t f = 0; f < faceCells_.size(); ++f) out[f] = psi[faceCells_[f]];
}

void ProcessorInterface::initInterfaceMatrixUpdate(const scalar* psi)
{
    // Gather own boundary values for this interface.
    for (std::size_t f = 0; f < faceCells_.size(); ++f) sendBuf_[f] = psi[faceCells_[f]];

    // Post receive then send (OpenFOAM nonblocking exchange ordering). Completion is a
    // single Pstream::waitAll() after all interfaces have posted, mirrors the matrix
    // interface loop's init-all / update-all structure.
    const int n = static_cast<int>(faceCells_.size());
    Pstream::irecv(recvBuf_.data(), n, nbrPart_, tag_);
    Pstream::isend(sendBuf_.data(), n, nbrPart_, tag_);
}

void ProcessorInterface::updateInterfaceMatrix(scalar* result, const scalar* coeffs)
{
    // recvBuf_ now holds the neighbour-side psi (post-wait). Apply the off-diagonal
    // coupling exactly as an internal face would, with the neighbour value exchanged
    // instead of read locally.
    for (std::size_t f = 0; f < faceCells_.size(); ++f)
    {
        result[faceCells_[f]] -= coeffs[f] * recvBuf_[f];
    }
}

} // namespace brae
