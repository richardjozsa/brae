#pragma once
// brae::buildLocalMesh, the decomposePar local fvMesh for ONE partition. From the global mesh +
// cell->partition map, build a complete local PrimitiveMesh (renumbered points/cells/faces) plus
// the processor-patch addressing. Mirrors OpenFOAM domainDecomposition / fvMeshDistribute:
//   - local internal faces: a global internal face with BOTH cells local (kept upper-triangular ,
//     global faces are owner-ascending and the global->local cell map is monotone);
//   - local boundary faces: a global boundary face whose cell is local (same patch);
//   - processor faces: a cut internal face -> a processor boundary patch toward the neighbour
//     partition. A cut face whose local cell is the global NEIGHBOUR is REVERSED so its area
//     vector points out of the local cell (as a boundary face must).
//   - both partitions of a (p,q) pair order their shared faces by GLOBAL face index, so face i on
//     p corresponds to face i on q with no communication.
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include <algorithm>
#include <array>
#include <map>
#include <string>
#include <vector>

namespace brae {

struct LocalMesh
{
    PrimitiveMesh                   mesh;            // the local mesh
    std::vector<label>              cellProcAddr;    // local cell -> global cell
    std::vector<label>              faceGlobal;      // local face -> global face (all local faces)
    std::vector<char>               faceFlip;        // local face reversed wrt the global face
    // One processor patch per neighbour partition (same order as the local mesh's processor patches):
    std::vector<int>                procNbr;         // neighbour partition
    std::vector<std::vector<label>> procFaceCells;   // local owner cell of each processor face
    std::vector<std::vector<label>> procGlobalFaces; // global face index (coeff lookup / matching)
    std::vector<std::vector<char>>  procOwnerSide;   // local cell is the global face's owner
};

inline LocalMesh buildLocalMesh(const PrimitiveMesh& gm, const std::vector<label>& cellToPart, int myPart)
{
    const label gNF = gm.nFaces(), gNIF = gm.nInternalFaces();
    const std::vector<label>& gOwn = gm.owner();
    const std::vector<label>& gNei = gm.neighbour();
    const std::vector<label>& gFV  = gm.faceVerts();
    const std::vector<label>& gFO  = gm.faceOffsets();
    const std::vector<PatchInfo>& gPatches = gm.patches();

    LocalMesh R;
    std::vector<label> g2lCell(gm.nCells(), -1);
    for (label c = 0; c < gm.nCells(); ++c)
        if (cellToPart[c] == myPart)
        {
            g2lCell[c] = static_cast<label>(R.cellProcAddr.size());
            R.cellProcAddr.push_back(c);
        }

    // Ordered list of local faces: {globalFace, reversed}. Internal first, then real boundary
    // patches, then processor patches. Local owner/neighbour built alongside.
    struct LF { label gf; bool rev; };
    std::vector<LF> lf;
    std::vector<label> lOwn, lNei;          // lNei only for internal faces
    std::vector<PatchInfo> lPatches;

    // (1) internal faces (both cells local).
    for (label f = 0; f < gNIF; ++f)
    {
        const label o = gOwn[f], n = gNei[f];
        if (cellToPart[o] == myPart && cellToPart[n] == myPart)
        {
            lf.push_back({f, false});
            lOwn.push_back(g2lCell[o]);
            lNei.push_back(g2lCell[n]);
        }
    }
    const label nLocalInternal = static_cast<label>(lf.size());

    // (2) real boundary patches (restricted to local cells), keeping the global patch order.
    for (const PatchInfo& P : gPatches)
    {
        const label start = static_cast<label>(lf.size());
        for (label i = 0; i < P.size; ++i)
        {
            const label f = P.start + i;
            if (cellToPart[gOwn[f]] == myPart)
            {
                lf.push_back({f, false});
                lOwn.push_back(g2lCell[gOwn[f]]);
            }
        }
        const label sz = static_cast<label>(lf.size()) - start;
        if (sz > 0) lPatches.push_back({P.name, P.type, start, sz});
    }

    // (3) processor patches: cut internal faces grouped by neighbour partition (sorted), faces by
    // global index. Tuple {globalFace, localCell, ownerSide}.
    std::map<int, std::vector<std::array<label, 3>>> byNbr;
    for (label f = 0; f < gNIF; ++f)
    {
        const label o = gOwn[f], n = gNei[f];
        const int po = cellToPart[o], pn = cellToPart[n];
        if (po == myPart && pn != myPart)      byNbr[pn].push_back({f, g2lCell[o], 1});
        else if (pn == myPart && po != myPart) byNbr[po].push_back({f, g2lCell[n], 0});
    }
    for (auto& kv : byNbr)
    {
        std::vector<std::array<label, 3>>& v = kv.second;
        std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a[0] < b[0]; });
        const label start = static_cast<label>(lf.size());
        std::vector<label> fc, gf;
        std::vector<char> os;
        for (const auto& t : v)
        {
            lf.push_back({t[0], t[2] == 0});   // reverse when the local cell is the neighbour
            lOwn.push_back(t[1]);
            fc.push_back(t[1]);
            gf.push_back(t[0]);
            os.push_back(static_cast<char>(t[2]));
        }
        lPatches.push_back({"procBoundary" + std::to_string(myPart) + "to" + std::to_string(kv.first), "processor",
                            start, static_cast<label>(v.size())});
        R.procNbr.push_back(kv.first);
        R.procFaceCells.push_back(std::move(fc));
        R.procGlobalFaces.push_back(std::move(gf));
        R.procOwnerSide.push_back(std::move(os));
    }

    // Renumber points: collect points used by local faces (ascending global index), map global->local.
    std::vector<label> usedPoint(gm.nPoints(), 0);
    for (const LF& e : lf)
        for (label k = gFO[e.gf]; k < gFO[e.gf + 1]; ++k) usedPoint[gFV[k]] = 1;
    std::vector<label> g2lPoint(gm.nPoints(), -1);
    std::vector<vector> lPoints;
    for (label p = 0; p < gm.nPoints(); ++p)
        if (usedPoint[p])
        {
            g2lPoint[p] = static_cast<label>(lPoints.size());
            lPoints.push_back(gm.points()[p]);
        }

    // Local face CSR (renumbered verts; reversed where flagged).
    std::vector<label> lFV, lFO(1, 0);
    for (const LF& e : lf)
    {
        const label a = gFO[e.gf], b = gFO[e.gf + 1];
        if (!e.rev) for (label k = a; k < b; ++k)       lFV.push_back(g2lPoint[gFV[k]]);
        else        for (label k = b - 1; k >= a; --k)  lFV.push_back(g2lPoint[gFV[k]]);
        lFO.push_back(static_cast<label>(lFV.size()));
    }

    // local face -> global face + flip (in local face order).
    R.faceGlobal.reserve(lf.size());
    R.faceFlip.reserve(lf.size());
    for (const LF& e : lf)
    {
        R.faceGlobal.push_back(e.gf);
        R.faceFlip.push_back(static_cast<char>(e.rev));
    }

    // Local owner/neighbour arrays: neighbour only for internal faces.
    std::vector<label> owner = lOwn;
    std::vector<label> neighbour(lNei.begin(), lNei.begin() + nLocalInternal);

    R.mesh.assign(std::move(lPoints), std::move(lFV), std::move(lFO),
                  std::move(owner), std::move(neighbour), std::move(lPatches),
                  static_cast<label>(R.cellProcAddr.size()));
    return R;
}

} // namespace brae
