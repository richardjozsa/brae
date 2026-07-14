#pragma once
// brae::distributeField, build a partition's local GeometricField<T> from the global field data and
// the local mesh (the decomposePar field scatter). Internal values are gathered by cellProcAddr;
// real boundary patches are restricted (uniform BCs copy directly; non-uniform restrict by the
// local->global face map); processor patches become coupled ProcessorFvPatchFields carrying the
// interpolation weights, so after evaluateBoundary their value() is the coupled face value and the
// serial fvc operators treat processor faces like internal faces.
#include "geometric_field.cuh"
#include "fv_patch.cuh"
#include "foam_field_reader.cuh"
#include "local_mesh.cuh"
#include <memory>
#include <stdexcept>
#include <vector>

namespace brae {

template <typename T>
GeometricField<T> distributeField(
    const FieldData<T>& gfd,
    const std::vector<PatchInfo>& gPatches,
    const LocalMesh& lm,
    const std::vector<FvPatch>& lpatches,
    const std::vector<std::vector<scalar>>& procWeights,
    int /*myRank*/)
{
    GeometricField<T> gf;
    gf.internal.resize(lm.mesh.nCells());
    if (gfd.internalUniform)
        for (auto& v : gf.internal)
            v = gfd.internalUniformValue;
    else
        for (label lc = 0; lc < lm.mesh.nCells(); ++lc)
            gf.internal[lc] = gfd.internalField[lm.cellProcAddr[lc]];

    std::size_t pj = 0;
    for (const FvPatch& p : lpatches)
    {
        if (p.type == "processor")
        {
            auto pf = std::make_unique<ProcessorFvPatchField<T>>(p, lm.procNbr[pj], 0);
            pf->setWeights(procWeights[pj]);
            gf.boundary.push_back(std::move(pf));
            ++pj;
            continue;
        }
        const PatchFieldData<T>* d = nullptr;
        for (const PatchFieldData<T>& b : gfd.boundary)   // literal
            if (b.name == p.name)
            {
                d = &b;
                break;
            }
        if (!d)
            for (const PatchFieldData<T>& b : gfd.boundary)   // else regex ((?i) ok)
                try
                {
                    if (std::regex_match(p.name, compileFoamRegex(b.name))) d = &b;
                }
                catch (const std::regex_error&) {}
        PatchFieldData<T> synth;                                                                    // constraint default
        if (!d && isConstraintPatchType(p.type))
        {
            synth.name = p.name;
            synth.type = p.type;
            d = &synth;
        }
        if (!d) throw std::runtime_error("distributeField: no boundaryField for patch " + p.name);

        if (d->valueUniform || d->values.empty())
        {
            gf.boundary.push_back(makePatchField<T>(p, *d));          // uniform / value-less: copy
        }
        else   // non-uniform: restrict by face map
        {
            label gStart = -1;
            for (const PatchInfo& gp : gPatches)
                if (gp.name == p.name)
                {
                    gStart = gp.start;
                    break;
                }
            PatchFieldData<T> rd = *d;
            rd.values.resize(p.size);
            for (label i = 0; i < p.size; ++i)
                rd.values[i] = d->values[lm.faceGlobal[p.start + i] - gStart];
            gf.boundary.push_back(makePatchField<T>(p, rd));
        }
    }
    return gf;
}

} // namespace brae
