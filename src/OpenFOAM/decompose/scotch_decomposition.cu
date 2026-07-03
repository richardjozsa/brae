#include "scotch_decomposition.cuh"

#include <cstdio>   // scotch.h uses FILE
#include <scotch.h>
#include <stdexcept>

namespace brae {

std::vector<label> scotchDecompose(const PrimitiveMesh& m, label nParts) {
    const label nC  = m.nCells();
    const label nIf = m.nInternalFaces();

    if (nParts <= 1) return std::vector<label>(nC, 0);

    // Cell dual graph in CSR: each internal face contributes two directed edges.
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();

    std::vector<SCOTCH_Num> xadj(nC + 1, 0);
    for (label f = 0; f < nIf; ++f) { ++xadj[own[f] + 1]; ++xadj[nei[f] + 1]; }
    for (label c = 0; c < nC; ++c) xadj[c + 1] += xadj[c];

    std::vector<SCOTCH_Num> adjncy(xadj[nC]);
    std::vector<SCOTCH_Num> cursor(xadj.begin(), xadj.end() - 1);  // running insert position
    for (label f = 0; f < nIf; ++f) {
        const label o = own[f], n = nei[f];
        adjncy[cursor[o]++] = n;
        adjncy[cursor[n]++] = o;
    }

    SCOTCH_Graph graph;
    if (SCOTCH_graphInit(&graph) != 0)
        throw std::runtime_error("SCOTCH_graphInit failed");

    if (SCOTCH_graphBuild(
            &graph,
            0,                              // baseval
            static_cast<SCOTCH_Num>(nC),    // vertnbr
            xadj.data(), xadj.data() + 1,   // verttab, vendtab (compact CSR)
            nullptr, nullptr,               // velotab (vertex wts), vlbltab
            static_cast<SCOTCH_Num>(adjncy.size()),
            adjncy.data(), nullptr) != 0)   // edgetab, edlotab (edge wts)
    {
        SCOTCH_graphExit(&graph);
        throw std::runtime_error("SCOTCH_graphBuild failed");
    }

    SCOTCH_Strat strat;
    SCOTCH_stratInit(&strat);

    std::vector<SCOTCH_Num> part(nC);
    const int rc = SCOTCH_graphPart(&graph, static_cast<SCOTCH_Num>(nParts), &strat, part.data());

    SCOTCH_stratExit(&strat);
    SCOTCH_graphExit(&graph);

    if (rc != 0) throw std::runtime_error("SCOTCH_graphPart failed");

    std::vector<label> cellToPart(nC);
    for (label c = 0; c < nC; ++c) cellToPart[c] = static_cast<label>(part[c]);
    return cellToPart;
}

} // namespace brae
