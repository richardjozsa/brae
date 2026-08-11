#pragma once
// functionObjects::scalarTransport -- OF src/functionObjects/solvers/scalarTransport.C.
//
// WHAT OF DOES, and what is reproduced here. OF's compressible branch is
//
//     fvm::ddt(rho, s) + fvm::div(phi, s, divScheme) - fvm::laplacian(D, s, laplacianScheme)
//         == fvOptions(rho, s)
//     sEqn.relax(relaxCoeff);  iterate until initialResidual() < tol
//
// On a STEADY solver the ddt term drops out, leaving div - laplacian, which is exactly the shape
// deviceSolveScalarTransport already assembles for k, epsilon, omega and the energy -- so the tracer
// goes through the case's own discretisation rather than a private one.
//
// LOOKED UP BY NAME, LATE. Everything this needs comes from the ObjectRegistry on FIRST EXECUTE, not
// from references captured at construction. That is OF's own arrangement (Time IS an objectRegistry;
// scalarTransport.C's transportedField() creates or finds its field on first use), and it is what makes
// construction order irrelevant: Time can be built at start-up -- so the functionObject report survives
// a later refusal -- while the solver it will drive is built 200 lines further down.
//
// NOT YET READY is not an error. If the registry has no solver at the first execute() the object simply
// defers; a steady solver registers before its loop, so this resolves on the first iteration.
//
// SCOPE, and what is refused rather than approximated:
//   * D: OF picks between a constant, a named nut field, and alphaD*nu + alphaDt*nut (scalarTransport.C
//     D()). ONLY the constant branch is built; the caller refuses the others BY NAME. Substituting a
//     constant for a nut-based diffusivity would change the answer while appearing to work.
//   * ddt: TRANSIENT is not supported. fvm::ddt(s) needs old/old2/ddt0 buffers (device_ddt.cuh:95) which
//     this object does not hold, so a transient solver must not register it -- that would solve a WRONG
//     equation, not an approximate one.
//
// PASSIVE: nothing here writes back into U, p, T, rho or the turbulence, so a case solves identically
// with the tracer present or absent. That is what makes it safe to run from the functionObject
// lifecycle rather than from inside the SIMPLE loop.

#include "brae_time.cuh"
#include "object_registry.cuh"
#include "device_simple_foam.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "fv_patch.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "brae_notice.cuh"
#include <filesystem>
#include <string>
#include <vector>

namespace brae {

class ScalarTransportFO : public FunctionObject
{
public:
    ScalarTransportFO(
        std::string name,
        std::string fieldName,
        std::string fieldPath,
        const ObjectRegistry& registry,
        scalar D,
        scalar relax,
        scalar tol)
      : name_(std::move(name)),
        fieldName_(std::move(fieldName)),
        fieldPath_(std::move(fieldPath)),
        registry_(registry),
        D_(D),
        relax_(relax),
        tol_(tol)
    {}

    const std::string& name() const override { return name_; }

    // OF solves the transport equation in execute(), every time step.
    bool execute() override
    {
        if (failed_) return true;                     // already reported; do not re-report each step
        if (!ready_ && !initialise()) return true;   // registry not populated yet; try again next step
        solver_->solvePassiveScalar(field_, boundary_, fieldName_.c_str(), Dbuf_, relax_, tol_);
        return true;
    }

    // Host copy for the writer, pulled only when a write is actually due -- the reason OF splits
    // write() from execute() -- so the tracer costs no device-to-host traffic per iteration.
    std::vector<scalar> hostField() const { return ready_ ? field_.host() : std::vector<scalar>(); }
    const std::string&  fieldName() const { return fieldName_; }
    bool                ready()     const { return ready_; }

private:
    // Resolve everything BY NAME, once. Returns false while the registry is still incomplete.
    bool initialise()
    {
        solver_ = registry_.lookupObject<DeviceSimpleSolver>("solver");
        const auto* fvp = registry_.lookupObject<std::vector<FvPatch>>("patches");
        const auto* m   = registry_.lookupObject<PrimitiveMesh>("mesh");
        if (!solver_ || !fvp || !m) return false;

        if (!std::filesystem::exists(fieldPath_))
        {
            noticeIgnored("functions/" + name_,
                          "scalarTransport field '" + fieldName_ + "' is not present at " + fieldPath_ +
                          ", so this tracer is NOT solved.");
            failed_ = true;
            return false;
        }

        const label nC = m->nCells();
        GeometricField<scalar> sf = buildField<scalar>(readField<scalar>(fieldPath_), *fvp, nC);
        sf.evaluateBoundary();
        field_.copyFrom(sf.internal);
        Dbuf_.copyFrom(std::vector<scalar>(static_cast<std::size_t>(nC), D_));

        const auto* g = registry_.lookupObject<FvGeometry>("geometry");
        if (!g) return false;
        boundary_ = buildDeviceBoundary(sf, *fvp, *g);

        ready_ = true;
        return true;
    }

    std::string           name_;
    std::string           fieldName_;
    std::string           fieldPath_;
    const ObjectRegistry& registry_;
    scalar                D_;
    scalar                relax_;
    scalar                tol_;

    DeviceSimpleSolver*   solver_ = nullptr;
    DeviceBuffer<scalar>  field_;
    DeviceBuffer<scalar>  Dbuf_;
    DeviceBoundary        boundary_;
    bool                  ready_  = false;
    bool                  failed_ = false;   // reported once; do not retry or re-report every step
};

}   // namespace brae
