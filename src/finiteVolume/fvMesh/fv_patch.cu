#include <stdexcept>
#include "fv_patch.cuh"
#include <cmath>

namespace brae {

std::vector<FvPatch> buildPatches(const PrimitiveMesh& m, const FvGeometry& g)
{
    std::vector<FvPatch> patches;
    patches.reserve(m.patches().size());
    for (const PatchInfo& pi : m.patches())
    {
        // OVERSET is not implemented, and it must not be mistaken for a constraint patch. It was listed in
        // isConstraintPatchType, so a field file omitting its boundaryField entry got a synthesised
        // constraint entry and the case RAN -- producing a converged, wrong answer with no message.
        //
        // Overset is not a boundary condition at all: OpenFOAM's src/overset replaces the matrix addressing
        // (fvMeshPrimitiveLduAddressing), because an acceptor cell's equation becomes an interpolation from
        // donor cells in another mesh region. Without hole cutting, a donor/acceptor search and that matrix
        // surgery, an overset mesh is just several disconnected meshes solved independently. Refuse.
        if (pi.type == "overset")
        {
            throw std::runtime_error(
                "brae: patch '" + pi.name + "' is type 'overset', and overset meshes are not supported. "
                "Overset replaces the matrix addressing (acceptor cells are interpolated from donors in "
                "another region), so running without it would solve the regions as if they were "
                "unconnected and converge to a wrong answer. See docs/roadmap.md.");
        }
        FvPatch p;
        p.name     = pi.name;
        p.type     = pi.type;
        p.inGroups = pi.inGroups;
        p.start    = pi.start;
        p.size     = pi.size;
        p.faceCells.resize(pi.size);
        p.deltaCoeffs.resize(pi.size);
        p.nf.resize(pi.size);
        p.magSf.resize(pi.size);
        p.Cf.resize(pi.size);
        for (label i = 0; i < pi.size; ++i)
        {
            const label f = pi.start + i;
            const label c = m.owner()[f];          // boundary face owner = adjacent cell
            p.faceCells[i] = c;
            // OF fvPatch::delta() is the normal-projected delta; deltaCoeffs = 1/(n.(Cf-C)).
            const vector nHat = g.Sf()[f] / g.magSf()[f];
            p.nf[i] = nHat;
            p.magSf[i] = g.magSf()[f];
            p.Cf[i] = g.Cf()[f];
            p.deltaCoeffs[i] = 1.0 / dot(g.Cf()[f] - g.C()[c], nHat);
        }
        patches.push_back(std::move(p));
    }
    return patches;
}

} // namespace brae
