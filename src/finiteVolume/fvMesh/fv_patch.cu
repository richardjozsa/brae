#include <stdexcept>
#include "fv_patch.cuh"
#include <cstdlib>
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
        // cyclicACMI: COUPLING VERIFIED ON A STATIC MESH, MOTION NOT. The interface itself now reproduces
        // OpenFOAM: on pimpleFoam/RAS/oscillatingInletACMI2D held static, laminar with Gauss upwind so
        // both codes solve the same equations, 10 steps of velocity agree to 4.7e-07 (L2) and the
        // per-face interface flux to 6e-08. The remaining refusal is about mesh MOTION, which is the one
        // thing an ACMI exists for: with the inlet channel sliding, the same case is at 3.0e-02.
        //
        // THE OLD JUSTIFICATION HERE WAS WRONG, and is written down because it cost a long hunt. It read
        // "contLocal ~0.33 where OF reaches 1e-14, concentrated on the cells touching the interface".
        // Both halves were the instrument: the residual omitted interfaceAddDiv, so it measured the
        // interface flux the pressure equation had just driven to zero. A per-cell continuity residual
        // could not have found any of the real defects anyway -- this interface fails by having both
        // sides balance cell-by-cell while transmitting different totals.
        //
        // What was actually wrong, all four found by tracing values against OpenFOAM term by term:
        //   1. the ACMI coverage was applied twice, in the face area AND in the interpolation weights
        //      (ami_interface.cuh)
        //   2. the mesh-move AMI rebuild threw away the interface flux (moveMesh)
        //   3. the coupled-patch flux was never written, so a restart rebuilt it from U and put 0.95%
        //      on the momentum diagonal of every source-side interface cell (foam_field_writer.cuh)
        //   4. cyc_/ami_.ifCoeff is shared between the momentum and pressure assemblies, so from the
        //      SECOND pressure corrector on, UEqn.H() read the pressure laplacian coefficient -- ~900x
        //      the momentum one (device_simple_foam.cu)
        //
        // A case that runs to a confident wrong answer is the one outcome this codebase does not accept,
        // so ACMI stays refused while the moving-mesh gap is open. BRAE_ALLOW_ACMI=1 opts in for
        // development; it is deliberately not a dictionary setting, so no case file can turn it on by
        // accident.
        if (pi.type == "cyclicACMI" && !std::getenv("BRAE_ALLOW_ACMI"))
            throw std::runtime_error(
                "brae: patch '" + pi.name + "' is type 'cyclicACMI'. The interface COUPLING now matches "
                "OpenFOAM -- on a static pimpleFoam/RAS/oscillatingInletACMI2D the velocity agrees to "
                "4.7e-07 and the per-face interface flux to 6e-08 -- but a cyclicACMI exists to slide, "
                "and with the mesh MOVING that case is still 3.0e-02 apart. Refused rather than solved "
                "to a plausible wrong answer. Set BRAE_ALLOW_ACMI=1 to run it anyway (development only).");

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
