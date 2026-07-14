#include "fv_patch.cuh"
#include <cmath>

namespace brae {

std::vector<FvPatch> buildPatches(const PrimitiveMesh& m, const FvGeometry& g)
{
    std::vector<FvPatch> patches;
    patches.reserve(m.patches().size());
    for (const PatchInfo& pi : m.patches())
    {
        FvPatch p;
        p.name     = pi.name;
        p.type     = pi.type;
        p.inGroups = pi.inGroups;
        p.start    = pi.start;
        p.size     = pi.size;
        p.faceCells.resize(pi.size);
        p.deltaCoeffs.resize(pi.size);
        p.nf.resize(pi.size);
        p.Cf.resize(pi.size);
        for (label i = 0; i < pi.size; ++i)
        {
            const label f = pi.start + i;
            const label c = m.owner()[f];          // boundary face owner = adjacent cell
            p.faceCells[i] = c;
            // OF fvPatch::delta() is the normal-projected delta; deltaCoeffs = 1/(n.(Cf-C)).
            const vector nHat = g.Sf()[f] / g.magSf()[f];
            p.nf[i] = nHat;
            p.Cf[i] = g.Cf()[f];
            p.deltaCoeffs[i] = 1.0 / dot(g.Cf()[f] - g.C()[c], nHat);
        }
        patches.push_back(std::move(p));
    }
    return patches;
}

} // namespace brae
