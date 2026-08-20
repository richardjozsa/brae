#pragma once
// _cpp REFERENCE -- OpenFOAM's fvOptions framework, plus explicitPorositySource/DarcyForchheimer.
//
// provenance:
//   openfoam:
//     fvOption.H / fvOptionList.C        the option list and its three hooks
//     cellSetOption.C                    selectionMode / cellZone / cellSet / all
//     explicitPorositySource.C:addSup    porosityEqn built, then `eqn -= porosityEqn`
//     porosityModel.C:addResistance      transformModelData then correct(UEqn)
//     DarcyForchheimer.C:correct         kinematic UEqn -> apply(..., one, nu, U)
//     DarcyForchheimerTemplates.C:apply  the resistance itself
//     DarcyForchheimer.C:calcTransformModelData   D = csys(diag(d)),  F = csys(diag(0.5*f))
//   brae:
//     reference: src/finiteVolume/cfdTools/general/fvOptions/fvOptions_cpp.cu
//     tests:     tests/test_fvoptions_cpp.cu, tests/fvoptions_vs_openfoam.sh
//
// THIS IS A FRAMEWORK PLUS ONE SOURCE, and the split is deliberate. ofscan counts 46 fv::option
// implementations; the framework (dictionary, cell selection, the three hooks UEqn.H and pEqn.H call)
// is shared by all of them, and each source is separate work. Reading the framework as "fvOptions is
// supported" would be exactly the silent-substitution failure this port exists to prevent, so an option
// whose `type` is not implemented is REFUSED BY NAME rather than skipped.
//
// THE THREE HOOKS, from simpleFoam's own text:
//     UEqn.H:11   == fvOptions(U)                 a source/sink on the momentum equation
//     UEqn.H:17   fvOptions.constrain(UEqn)       setValues-style constraints
//     UEqn.H:23   fvOptions.correct(U)            post-solve field manipulation
// Only the first is implemented here; the other two are refused when an option needs them.
//
// THE SIGN, which passes through two negations and is worth stating once. explicitPorositySource builds
// a porosityEqn and does `eqn -= porosityEqn` into the fvOptions matrix; simpleFoam then writes
// `UEqn == fvOptions(U)`, i.e. UEqn - optionsEqn. The two negations cancel, so the NET effect on the
// momentum matrix is the porosity equation as written:
//     diag[c]   += V[c]*isoCd
//     source[c] -= V[c]*((Cd - I*isoCd) & U[c])
// A port that applied only one of the negations gets a porosity that ACCELERATES the flow.
#include "cf_types.cuh"
#include "foam_dict.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "ldu_matrix.cuh"
#include <string>
#include <vector>

namespace brae {
namespace cpu {
namespace fvOptions {

// One fv::option. Only explicitPorositySource/DarcyForchheimer is implemented; `unsupported` carries the
// type name of anything else so the caller can refuse it by name.
struct Option
{
    // A rotorDiskSource: implemented, but its parameters and geometry come from readFvOptions
    // and the mesh rather than from this list. Recorded so the driver knows one is present.
    bool rotorDisk = false;

    std::string        name;
    std::string        type;
    bool               active = true;
    std::vector<label> cells;                 // resolved by cellSetOption's rules; empty => all cells
    bool               allCells = false;
    std::string        unsupported;           // non-empty => this option's type is not implemented

    // DarcyForchheimer, already transformed into the global frame with the 0.5 folded into F.
    tensor D{};
    tensor F{};
};

struct OptionList
{
    std::vector<Option> options;
    bool empty() const { return options.empty(); }
    // The first option whose type this port does not implement, or "" when all are implemented.
    std::string firstUnsupported() const;
};

// Read system/fvOptions or constant/fvOptions (OpenFOAM looks in both). Absent file => empty list.
OptionList read(const std::string& caseDir, const PrimitiveMesh& m);

// UEqn.H's `== fvOptions(U)`, for a KINEMATIC momentum equation (nu, not mu -- DarcyForchheimer.C
// dispatches on UEqn.dimensions() and takes the `one`/nu branch when the equation is not in force units).
void addSup(
    const OptionList&             opts,
    FvVectorMatrix&               UEqn,
    const GeometricField<vector>& U,
    scalar                        nu,
    const FvGeometry&             g);

} // namespace fvOptions
} // namespace cpu
} // namespace brae
