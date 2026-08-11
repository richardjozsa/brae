#pragma once
// functionObjects::scalarTransport -- OF src/functionObjects/solvers/scalarTransport.C.
//
// Lives here, next to DeviceSimpleSolver, rather than in OpenFOAM/db/Time: the solve runs on the DEVICE
// against the flux the solver already holds, so building it needs the solver type. brae_time.cuh keeps
// the lifecycle and a factory table (the analogue of OF's addToRunTimeSelectionTable); the owning solver
// registers this type before FunctionObjectList::read().
//
// WHAT OF DOES, and what is reproduced here. OF's compressible branch is
//
//     fvm::ddt(rho, s) + fvm::div(phi, s, divScheme) - fvm::laplacian(D, s, laplacianScheme)
//         == fvOptions(rho, s)
//     sEqn.relax(relaxCoeff);  iterate until initialResidual() < tol
//
// On a STEADY solver the ddt term drops out, leaving div - laplacian, which is exactly the shape
// deviceSolveScalarTransport already assembles for k, epsilon, omega and the energy. The tracer
// therefore goes through the same discretisation path the rest of the case does rather than a private
// one.
//
// D: OF picks between a constant, a named nut field, and alphaD*nu + alphaDt*nut (scalarTransport.C
// D(), constantD_ / nutName_ / else). ONLY the constant branch is implemented. The other two are
// refused by name at construction rather than silently replaced by a constant -- substituting a
// different diffusivity would change the answer while looking like it worked.
//
// PASSIVE: nothing here writes back into U, p, T, rho or the turbulence, so a case solves identically
// with the tracer present or absent. That is what makes it safe to run from the functionObject
// lifecycle rather than from inside the SIMPLE loop.

#include "brae_time.cuh"
#include "device_simple_foam.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "foam_field_writer.cuh"
#include "fv_patch.cuh"
#include <memory>
#include <string>
#include <vector>

namespace brae {

class ScalarTransportFO : public FunctionObject
{
public:
    ScalarTransportFO(
        std::string name,
        std::string fieldName,
        DeviceSimpleSolver& solver,
        DeviceBuffer<scalar> field,
        DeviceBoundary boundary,
        scalar D,
        scalar relax,
        scalar tol,
        label nCells)
      : name_(std::move(name)),
        fieldName_(std::move(fieldName)),
        solver_(solver),
        field_(std::move(field)),
        boundary_(std::move(boundary)),
        relax_(relax),
        tol_(tol)
    {
        // Built ONCE: a uniform diffusivity re-uploaded every step would be pure waste.
        D_.copyFrom(std::vector<scalar>(static_cast<std::size_t>(nCells), D));
    }

    const std::string& name() const override { return name_; }

    // OF solves the transport equation in execute(), every time step.
    bool execute() override
    {
        solver_.solvePassiveScalar(field_, boundary_, fieldName_.c_str(), D_, relax_, tol_);
        return true;
    }

    // The host copy for the writer. Pulled only when a write is actually due -- the whole point of OF
    // splitting write() from execute() -- so the tracer costs no device-to-host traffic per iteration.
    std::vector<scalar> hostField() const { return field_.host(); }
    const std::string& fieldName() const { return fieldName_; }

private:
    std::string name_;
    std::string fieldName_;
    DeviceSimpleSolver& solver_;
    DeviceBuffer<scalar> field_;
    DeviceBoundary boundary_;
    DeviceBuffer<scalar> D_;
    scalar relax_;
    scalar tol_;
};

}   // namespace brae
