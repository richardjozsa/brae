#pragma once
// brae::DomainDecomposition, the decomposePar-equivalent local view for ONE partition.
// Mirrors OpenFOAM fvMeshDistribute / processorPolyPatch construction: from the global
// mesh + cell->part map, build this partition's local cell numbering (cellProcAddressing)
// and its processor interfaces. An internal global face whose two cells land on different
// partitions becomes a processor-interface face, the SAME coupling as an internal face,
// with the neighbour value delivered by exchange.
//
// Matched ordering without communication: both partitions of a (p,q) pair order their
// shared faces by GLOBAL face index, so face i on p corresponds to face i on q.
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include "processor_interface.cuh"
#include <vector>

namespace brae {

class DomainDecomposition {
public:
    DomainDecomposition(const PrimitiveMesh& global,
                        const std::vector<label>& cellToPart,
                        int myPart);

    label nLocalCells()         const { return static_cast<label>(cellProcAddr_.size()); }
    label nLocalInternalFaces() const { return nLocalInternal_; }

    const std::vector<label>& cellProcAddressing() const { return cellProcAddr_; }  // local -> global
    std::vector<ProcessorInterface>& interfaces()        { return ifaces_; }

    // Local internal-face addressing (upper-triangular, local cell indices) and the global face
    // each maps to, lets a global lduMatrix be distributed to this partition by index lookup.
    const std::vector<label>& localOwner()      const { return localOwner_; }
    const std::vector<label>& localNeighbour()  const { return localNeighbour_; }
    const std::vector<label>& localFaceGlobal() const { return localFaceGlobal_; }

    // Per interface (same order as interfaces()): the global cut-face index and whether the local
    // cell is that face's OWNER, needed to pull the off-diagonal coeff (upper vs lower) for the
    // interface boundary coefficients.
    const std::vector<std::vector<label>>& interfaceGlobalFaces() const { return ifaceGlobalFaces_; }
    const std::vector<std::vector<char>>&  interfaceOwnerSide()   const { return ifaceOwnerSide_; }

    // For validation: the global cell index on the far side of each interface face, in the
    // same order as that interface's faceCells. The exchange must deliver exactly these.
    const std::vector<std::vector<label>>& nbrGlobalCells() const { return nbrGlobalCells_; }

private:
    std::vector<label>              cellProcAddr_;
    std::vector<ProcessorInterface> ifaces_;
    std::vector<std::vector<label>> nbrGlobalCells_;
    std::vector<label>              localOwner_, localNeighbour_, localFaceGlobal_;
    std::vector<std::vector<label>> ifaceGlobalFaces_;
    std::vector<std::vector<char>>  ifaceOwnerSide_;
    label                           nLocalInternal_ = 0;
};

} // namespace brae
