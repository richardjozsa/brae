#include "scotch_decomposition.cuh"

#include <cstdio>   // scotch.h uses FILE
#include <scotch.h>
#include <stdexcept>
#include <fstream>
#include <filesystem>
#include <string>

namespace brae {

std::vector<label> scotchDecompose(const PrimitiveMesh& m, label nParts)
{
    const label nC  = m.nCells();
    const label nIf = m.nInternalFaces();

    if (nParts <= 1) return std::vector<label>(nC, 0);

    // Cell dual graph in CSR: each internal face contributes two directed edges.
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();

    std::vector<SCOTCH_Num> xadj(nC + 1, 0);
    for (label f = 0; f < nIf; ++f)
    {
        ++xadj[own[f] + 1];
        ++xadj[nei[f] + 1];
    }
    for (label c = 0; c < nC; ++c) xadj[c + 1] += xadj[c];

    std::vector<SCOTCH_Num> adjncy(xadj[nC]);
    std::vector<SCOTCH_Num> cursor(xadj.begin(), xadj.end() - 1);  // running insert position
    for (label f = 0; f < nIf; ++f)
    {
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

// Cached SCOTCH decomposition: the cell->partition map is STATIC per (mesh, nParts), but scotchDecompose costs
// ~1s on a 1M-cell mesh and ran on every launch (a top-3 startup cost). Persist it to
// cacheDir/.brae_decomp_np<N> and reload when the file is at least as new as cacheDir/owner (mesh unchanged) and
// the header (magic, nParts, nCells) matches; otherwise recompute and write. Falls back to a plain decompose on
// any I/O problem, so it can never be wrong -- at worst it recomputes.
std::vector<label> scotchDecomposeCached(const PrimitiveMesh& m, label nParts, const std::string& cacheDir)
{
    namespace fs = std::filesystem;
    constexpr unsigned kMagic = 0x44454331u;                 // "DEC1"
    const label nC = m.nCells();
    const std::string cachePath = cacheDir + "/.brae_decomp_np" + std::to_string(nParts);
    const std::string ownerPath = cacheDir + "/owner";

    std::error_code ec;
    const bool fresh = fs::exists(cachePath, ec) && fs::exists(ownerPath, ec)
                    && fs::last_write_time(cachePath, ec) >= fs::last_write_time(ownerPath, ec);
    if (fresh)
    {
        std::ifstream is(cachePath, std::ios::binary);
        unsigned magic = 0; label np = 0, n = 0;
        is.read(reinterpret_cast<char*>(&magic), sizeof magic);
        is.read(reinterpret_cast<char*>(&np),    sizeof np);
        is.read(reinterpret_cast<char*>(&n),     sizeof n);
        if (is && magic == kMagic && np == nParts && n == nC)
        {
            std::vector<label> v(nC);
            is.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(nC) * sizeof(label));
            if (is) return v;                                // cache hit
        }
    }

    std::vector<label> v = scotchDecompose(m, nParts);       // miss -> compute + persist
    std::ofstream os(cachePath, std::ios::binary);
    if (os)
    {
        const unsigned magic = kMagic; const label np = nParts, n = nC;
        os.write(reinterpret_cast<const char*>(&magic), sizeof magic);
        os.write(reinterpret_cast<const char*>(&np),    sizeof np);
        os.write(reinterpret_cast<const char*>(&n),     sizeof n);
        os.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(nC) * sizeof(label));
    }
    return v;
}

} // namespace brae
