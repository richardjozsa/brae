#pragma once
// brae::buildCyclicField, assemble a GeometricField<T> on a mesh with cyclic patches: cyclic patches get
// a CyclicFvPatchField (coupled to the local neighbour cells), empty patches an EmptyPatchField, all
// others zeroGradient. Mirrors how distributeFromCells builds a processor-coupled field, but local.
#include "geometric_field.cuh"
#include "fv_patch_field.cuh"
#include "fv_patch.cuh"
#include "cyclic_interface.cuh"
#include <memory>
#include <vector>

namespace brae {

// wallNoSlip: give "wall" patches a NoSlipPatchField (value 0), used for velocity-like fields so the
// fvm operators get the correct wall coefficients. Scalars (p, rAU) keep zeroGradient walls.
template <typename T>
GeometricField<T> buildCyclicField(const std::vector<T>& cells, const std::vector<FvPatch>& fvp,
                                   const std::vector<CyclicInterface>& cyclics, bool wallNoSlip = false) {
    GeometricField<T> gf; gf.internal = cells;
    for (label pi = 0; pi < (label)fvp.size(); ++pi) {
        const FvPatch& p = fvp[pi];
        const CyclicInterface* ci = nullptr;
        for (const CyclicInterface& c : cyclics) if (c.patch == pi) { ci = &c; break; }
        if (ci) {
            auto pf = std::make_unique<CyclicFvPatchField<T>>(p, ci->nbrFaceCells, ci->weights);
            if (!ci->translational) pf->setTransform(ci->forwardT);   // vectors rotate; scalars ignore it
            gf.boundary.push_back(std::move(pf));
        }
        else if (p.type == "empty")        gf.boundary.push_back(std::make_unique<EmptyPatchField<T>>(p));
        else if (wallNoSlip && p.type == "wall") gf.boundary.push_back(std::make_unique<NoSlipPatchField<T>>(p));
        else                               gf.boundary.push_back(std::make_unique<ZeroGradientPatchField<T>>(p));
    }
    gf.evaluateBoundary();
    return gf;
}

} // namespace brae
