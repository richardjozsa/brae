#include "domain_decomposition.cuh"
#include <algorithm>
#include <array>
#include <map>

namespace brae {

DomainDecomposition::DomainDecomposition(const PrimitiveMesh& global,
                                         const std::vector<label>& cellToPart,
                                         int myPart) {
    const label nC  = global.nCells();
    const label nIf = global.nInternalFaces();
    const std::vector<label>& own = global.owner();
    const std::vector<label>& nei = global.neighbour();

    // Local cell numbering: global cells of this partition, in ascending global order.
    std::vector<label> globalToLocal(nC, -1);
    for (label c = 0; c < nC; ++c) {
        if (cellToPart[c] == myPart) {
            globalToLocal[c] = static_cast<label>(cellProcAddr_.size());
            cellProcAddr_.push_back(c);
        }
    }

    // Classify internal global faces. A local-local face joins this partition's local internal-face
    // addressing; a cut face -> a processor-interface entry, grouped by neighbour partition.
    // Interface tuple: {globalFace, localFaceCell, nbrGlobalCell, localCellIsOwner}.
    std::map<int, std::vector<std::array<label, 4>>> byNbr;
    for (label f = 0; f < nIf; ++f) {
        const label o = own[f], n = nei[f];
        const int po = cellToPart[o], pn = cellToPart[n];
        if (po == myPart && pn == myPart) {
            localOwner_.push_back(globalToLocal[o]);      // global faces are owner-ascending and
            localNeighbour_.push_back(globalToLocal[n]);  // globalToLocal is monotone -> still upper-tri
            localFaceGlobal_.push_back(f);
            ++nLocalInternal_;
        } else if (po == myPart) {
            byNbr[pn].push_back({f, globalToLocal[o], n, 1});   // local cell is the owner
        } else if (pn == myPart) {
            byNbr[po].push_back({f, globalToLocal[n], o, 0});   // local cell is the neighbour
        }
    }

    // One ProcessorInterface per neighbour partition, faces ordered by global face index.
    for (auto& kv : byNbr) {
        std::vector<std::array<label, 4>>& v = kv.second;
        std::sort(v.begin(), v.end(),
                  [](const std::array<label, 4>& a, const std::array<label, 4>& b) {
                      return a[0] < b[0];
                  });

        std::vector<label> faceCells, nbrG, gFaces;
        std::vector<char>  ownerSide;
        faceCells.reserve(v.size()); nbrG.reserve(v.size()); gFaces.reserve(v.size()); ownerSide.reserve(v.size());
        for (const std::array<label, 4>& t : v) {
            gFaces.push_back(t[0]); faceCells.push_back(t[1]); nbrG.push_back(t[2]); ownerSide.push_back(static_cast<char>(t[3]));
        }

        ifaces_.emplace_back(myPart, kv.first, std::move(faceCells));
        nbrGlobalCells_.push_back(std::move(nbrG));
        ifaceGlobalFaces_.push_back(std::move(gFaces));
        ifaceOwnerSide_.push_back(std::move(ownerSide));
    }
}

} // namespace brae
