#pragma once
// brae::cellWallDist, OpenFOAM wallDist::New(mesh).y(): the cell-wise wall-distance field (every cell),
// used by kOmegaSST F1/F2/F3 and Spalart-Allmaras dTilda.
//
// Ported byte-for-byte from OF's DEFAULT method, meshWave (wallDist.H "method meshWave" ->
// patchDistMethods::meshWave -> patchWave -> FaceCellWave<wallPoint>):
//   1. SEED   (patchWave::setChangedFaces, patchWave.C:64): every wall-patch face f gets
//             faceInfo[f] = wallPoint(faceCentre Cf[f], distSqr 0). These are the changedFaces.
//   2. WAVE   (FaceCellWave::faceToCell / cellToFace): the front propagates origin (a wall-face CENTRE)
//             cell<->face through the mesh graph. wallPoint::update (wallPointI.H): dist2 = magSqr(pt-origin);
//             accept iff first-visit, OR strictly closer by more than propagationTol_ (= 0.01, the OF default
//             "just to limit propagation of small changes"). Each cell converges to the nearest PROPAGATED
//             wall-face centre, NOT a global Euclidean min; the front is connectivity-dependent, exactly as OF.
//   3. y      (patchWave::getValues, patchWave.C:99): y[c] = sqrt(cellInfo[c].distSqr()).
//   4. correctWalls (default true, patchWave.C:203): boundary-adjacent cells are OVERWRITTEN with the EXACT
//             near-wall distance, correctBoundaryFaceCells (point-to-face-polygon for a cell's own wall faces)
//             and correctBoundaryPointCells (point-connected). brae::nearWallDist computes exactly that and is
//             validated machine-precision vs OF's correctWalls on the gated near-wall cells (the SST-relevant
//             region). Interior cells keep the wave's face-centre distance (OF's own meshWave overestimate);
//             there F1/F2 -> 0 so the value cannot move the model (ctest treats interior as informational).
//
// This is O(nCells * frontRevisits), the FaceCellWave front, NOT the O(nCells * nWallFaces) brute force it
// replaces; on complex geometries (motorBike: ~40k wall faces) that is the difference between a multi-minute
// startup and seconds. Static geometry -> computed ONCE at setup.
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "near_wall_dist.cuh"   // closestPointOnTriangle, pointToFaceDist, nwdGreat (= OF correctWalls)
#include <vector>
#include <cmath>

namespace brae {

inline std::vector<scalar> cellWallDist(
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches)
{
    const std::vector<vector>& C   = g.C();
    const std::vector<vector>& Cf  = g.Cf();
    const std::vector<label>&  own = m.owner();
    const std::vector<label>&  nei = m.neighbour();
    const label nCells = m.nCells();
    const label nFaces = m.nFaces();
    const label nIntF  = m.nInternalFaces();

    std::vector<scalar> y(nCells, nwdGreat);                  // OF wallDist: y initialised to GREAT

    // the wave seed: every wall-patch face (patchWave::setChangedFaces)
    std::vector<char> isWallFace(nFaces, 0);
    bool anyWall = false;
    for (const FvPatch& p : patches)
        if (p.type == "wall" && p.size > 0)
        {
            anyWall = true;
            for (label i = 0; i < p.size; ++i) isWallFace[p.start + i] = 1;
        }
    if (!anyWall) return y;                                   // no walls -> all GREAT (no SST near-wall branch)

    // cell -> faces adjacency (OF mesh_.cells()), CSR built from owner/neighbour
    std::vector<label> cfOff(nCells + 1, 0);
    for (label f = 0; f < nFaces; ++f) cfOff[own[f] + 1]++;
    for (label f = 0; f < nIntF;  ++f) cfOff[nei[f] + 1]++;
    for (label c = 0; c < nCells; ++c) cfOff[c + 1] += cfOff[c];
    std::vector<label> cfList(cfOff[nCells]);
    {
        std::vector<label> cur(cfOff.begin(), cfOff.end() - 1);
        for (label f = 0; f < nFaces; ++f) cfList[cur[own[f]]++] = f;
        for (label f = 0; f < nIntF;  ++f) cfList[cur[nei[f]]++] = f;
    }

    // FaceCellWave<wallPoint> front. origin = wall-face centre; distSqr = magSqr(pt - origin).
    const scalar tol = 0.01;                                  // OF FaceCellWave::propagationTol_
    std::vector<vector> faceOrg(nFaces), cellOrg(nCells);
    std::vector<scalar> faceD2(nFaces, 0.0), cellD2(nCells, 0.0);
    std::vector<char> faceSet(nFaces, 0), cellSet(nCells, 0);
    std::vector<char> faceQ(nFaces, 0), cellQ(nCells, 0);     // already-in-changed-list flags (OF changedFace_/changedCell_)

    // wallPoint::update (wallPointI.H): first visit accepts any value; else accept iff strictly closer by > tol.
    auto update = [&](char& set, scalar& d2cur, vector& orgcur, const vector& pt, const vector& org) -> bool
    {
        const scalar d2 = magSqr(pt - org);
        if (!set)
        {
            d2cur = d2;
            orgcur = org;
            set = 1;
            return true;
        }
        const scalar diff = d2cur - d2;
        if (diff < 0) return false;                                              // already nearer
        if (diff < 1e-300 || (d2cur > 1e-300 && diff / d2cur < tol)) return false; // improvement too small
        d2cur = d2;
        orgcur = org;
        return true;
    };

    std::vector<label> changedFaces, changedCells;
    changedFaces.reserve(nFaces / 8 + 1);
    for (label f = 0; f < nFaces; ++f)
        if (isWallFace[f])
        {
            faceOrg[f] = Cf[f];
            faceD2[f] = 0.0;
            faceSet[f] = 1;
            faceQ[f] = 1;
            changedFaces.push_back(f);
        }

    while (!changedFaces.empty())
    {
        changedCells.clear();                                // faceToCell
        for (label f : changedFaces)
        {
            faceQ[f] = 0;
            const vector org = faceOrg[f];
            const label o = own[f];
            if (update(cellSet[o], cellD2[o], cellOrg[o], C[o], org) && !cellQ[o])
            {
                cellQ[o] = 1;
                changedCells.push_back(o);
            }
            if (f < nIntF)
            {
                const label n = nei[f];
                if (update(cellSet[n], cellD2[n], cellOrg[n], C[n], org) && !cellQ[n])
                {
                    cellQ[n] = 1;
                    changedCells.push_back(n);
                }
            }
        }
        changedFaces.clear();                                // cellToFace
        for (label c : changedCells)
        {
            cellQ[c] = 0;
            const vector org = cellOrg[c];
            for (label k = cfOff[c]; k < cfOff[c + 1]; ++k)
            {
                const label f = cfList[k];
                if (update(faceSet[f], faceD2[f], faceOrg[f], Cf[f], org) && !faceQ[f])
                {
                    faceQ[f] = 1;
                    changedFaces.push_back(f);
                }
            }
        }
    }
    for (label c = 0; c < nCells; ++c)
        if (cellSet[c]) y[c] = std::sqrt(cellD2[c]);

    // correctWalls = correctBoundaryFaceCells THEN correctBoundaryPointCells (cellDistFuncs.C), in that order:
    // a cell already corrected by its own wall face is NOT overwritten by the point pass.
    const std::vector<vector>& pts = m.points();
    const std::vector<label>&  fv  = m.faceVerts();
    const std::vector<label>&  fo  = m.faceOffsets();

    // (1) correctBoundaryFaceCells: a cell owning wall face(s) -> exact point-to-face-polygon dist to its own
    // wall faces (min over them). brae::nearWallDist is the same exact routine OF's smallestDist uses.
    const std::vector<std::vector<scalar>> nwd = nearWallDist(m, g, patches);
    std::vector<scalar> wallY(nCells, nwdGreat);
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
        if (patches[pi].type == "wall")
            for (label i = 0; i < patches[pi].size; ++i)
            {
                const label c = patches[pi].faceCells[i];
                if (nwd[pi][i] < wallY[c]) wallY[c] = nwd[pi][i];     // min over the cell's wall faces
            }

    // (2) correctBoundaryPointCells: a cell point-connected to a wall vertex (and NOT already face-corrected)
    // -> exact dist to the wall faces sharing that vertex (OF pointFaces[patchPointi]), min over them.
    std::vector<char> isWallVtx(pts.size(), 0);
    std::vector<std::vector<label>> vtxWallFaces(pts.size());          // wall vertex -> incident wall faces (global idx)
    for (const FvPatch& p : patches)
        if (p.type == "wall")
            for (label i = 0; i < p.size; ++i)
            {
                const label gf = p.start + i;
                for (label j = fo[gf]; j < fo[gf + 1]; ++j)
                {
                    const label v = fv[j];
                    isWallVtx[v] = 1;
                    vtxWallFaces[v].push_back(gf);
                }
            }
    std::vector<scalar> ptY(nCells, nwdGreat);
    for (label f = 0; f < nFaces; ++f)
    {
        const label c0 = own[f], c1 = (f < nIntF ? nei[f] : -1);
        for (label j = fo[f]; j < fo[f + 1]; ++j)
        {
            const label v = fv[j];
            if (!isWallVtx[v]) continue;
            for (const label wf : vtxWallFaces[v])
            {
                if (wallY[c0] >= nwdGreat) ptY[c0] = std::fmin(ptY[c0], pointToFaceDist(C[c0], pts, fv, fo[wf], fo[wf + 1]));
                if (c1 >= 0 && wallY[c1] >= nwdGreat) ptY[c1] = std::fmin(ptY[c1], pointToFaceDist(C[c1], pts, fv, fo[wf], fo[wf + 1]));
            }
        }
    }

    for (label c = 0; c < nCells; ++c)
    {
        if (wallY[c] < nwdGreat)      y[c] = wallY[c];               // face-corrected (priority)
        else if (ptY[c] < nwdGreat)   y[c] = ptY[c];                // point-corrected
    }
    return y;
}

} // namespace brae
