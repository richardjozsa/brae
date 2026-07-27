#pragma once
// cf GPU offload (#4): the device-resident SIMPLE(+kEpsilon) solver as a reusable object. U/p/phi/k/eps/nut
// all live on the GPU between iterations; step() runs one SIMPLE iteration entirely on device:
//   faithful momentum  div(phi,U) - laplacian(nuEff,U) - fvc::div(nuEff*dev2(T(grad U)))   (BiCGStab/comp)
//   AMG-PCG pressure correction + conservative flux + corrector
//   (turbulent) device kEpsilon::correct()  [production -> wall functions -> eps/k -> nut]
// The loop body is the one validated machine-precision vs the CPU oracle in test_gpu_resident_turb. The
// AMG agglomeration is built once (static geometry); the coarse matrix is re-Galerkined each step (rAU
// changes). step() returns the OF residualControl signal (initial residuals of the U and p solves).
//
// nuEff at boundary faces = the adjacent-cell value (this path's convention, consistent with its laplacian
// boundary). The fully-faithful boundary-nut (nutkWallFunction) treatment lives in the host simple_foam.cuh.
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fvc.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "device_ldu.cuh"
#include "device_blas.cuh"
#include "device_pcg.cuh"
#include "device_simple.cuh"
#include "device_boundary.cuh"
#include "device_cyclic.cuh"
#include "device_ami.cuh"
#include "device_interface.cuh"   // interface<Op>() overloads dispatching to the cyclic/AMI backends
#include "device_kepsilon.cuh"
#include "device_komega_sst.cuh"   // deviceS2/deviceF2/deviceNutSST for the startup validate() correctNut
#include "device_smagorinsky.cuh"  // deviceSmagorinskyNut for the pure-LES (algebraic sub-grid nut) path
#include "cell_wall_dist.cuh"
#include "cell_max_delta.cuh"      // cellMaxDeltaXYZ -> hmax_ (SA-IDDES filter width, maxDeltaxyz)
#include "device_mrf.cuh"
#include "device_divdevreff.cuh"
#include "device_ddt.cuh"          // transient fvm::ddt(U) for the PIMPLE path (no-op in steady SIMPLE)
#include "device_amg.cuh"
#include "fv_options.cuh"
#include "device_fvoptions.cuh"
#include "solver_controls.cuh"   // NutWall, DeviceSimpleControls, DeviceSimpleResidual (moved out for reuse)
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

namespace brae {

// NutWall, DeviceSimpleControls, DeviceSimpleResidual now live in solver_controls.cuh (included above).

class DeviceSimpleSolver
{
public:
    DeviceSimpleSolver(
        const PrimitiveMesh& m,
        const FvGeometry& g,
        const std::vector<FvPatch>& fvp,
        const GeometricField<vector>& U,
        const GeometricField<scalar>& p,
        const SurfaceScalarField& phi,
        const DeviceSimpleControls& ctl,
        const GeometricField<scalar>* k = nullptr,
        const GeometricField<scalar>* eps = nullptr,
        const GeometricField<scalar>* nut = nullptr,
        const GeometricField<scalar>* ReThetat = nullptr,
        const GeometricField<scalar>* gammaInt = nullptr);

    // OF turbulence-model load sequence, ported byte-for-byte (do NOT skip, this is why OF never blows up on a
    // case cf does): (1) the model ctor bounds the read fields  [kEpsilon.C:105-106 bound(k_,kMin_); bound(epsilon_,...);
    // kOmegaSSTBase.C:200-201 bound(k_,kMin_); bound(omega_,omegaMin_)]  and (2) simpleFoam.C:92 turbulence->validate()
    // -> eddyViscosity::validate() -> correctNut(), recomputing the INTERNAL nut from the bounded k/(omega|eps) BEFORE
    // iter 1. Without (2) the FIRST momentum predictor runs with nut straight from 0/nut, which is `uniform 0` in
    // motorBike AND pitzDaily, i.e. fully laminar; harmless on attached flows but the seed of divergence on high-Re
    // skewed meshes. kMin_/epsilonMin_/omegaMin_ default to SMALL (= 1e-15), matching the floors used in correct().
    void validateTurbulence();
    // PIMPLE-foundation composable phase: turbulence transport (k/eps/omega/nuTilda -> nut). Uses only members,
    // so SIMPLE's step() and a future PIMPLE outer loop both call it (once per outer corrector) unchanged.
    void correctTurbulence();
    // PIMPLE-foundation composable phase 1: the momentum predictor -- assemble div(phi,U) - laplacian(nuEff,U)
    // - fvc::div(nuEff dev2(grad U)^T) + fvOptions/MRF/limiters + body force, relax, and solve each U component
    // (BiCGStab/GaussSeidel). Fills Uk_ + the shared members mDiagR/mUp/mLo/iC/bCb/relaxSrc (the relaxed matrix
    // + boundary coeffs + relaxed source) that correctPressureVelocity() reads. res gets the U-solve residuals.
    void solveMomentumPredictor(DeviceSimpleResidual& res);
    // PIMPLE-foundation composable phase: pressure-velocity coupling (SIMPLE/SIMPLEC corrector). Reads the
    // predictor outputs (member relaxed matrix + boundary coeffs + source), forms rAU/HbyA, solves the pressure
    // correction (with non-orth passes) and corrects U + the conservative face flux. res gets the p residuals.
    void correctPressureVelocity(DeviceSimpleResidual& res);

    DeviceSimpleResidual step();

    // --- Transient (PIMPLE) interface -- reuses the three composable phases above under an outer/inner-corrector loop,
    // with the implicit fvm::ddt(U) folded into solveMomentumPredictor. setDdtScheme() switches this solver from steady
    // (SIMPLE, the default) to transient; the steady step() path is completely unaffected (ddt stays a no-op).
    void setDdtScheme(DdtScheme s) { ddtScheme_ = s; }
    // Advance one time level: store U.oldTime()[.oldTime()] + roll deltaT -> deltaT0 (OF runTime++ / GeometricField::oldTime).
    void advanceTime(scalar deltaT);
    // One PIMPLE time step: advanceTime, then nOuterCorrectors x { momentum predictor (ddt folded in); nCorrectors x
    // pressure-velocity correction; turbulence }, then continuity errors. Returns the outer-loop residual signal.
    DeviceSimpleResidual pimpleStep(scalar deltaT, int nOuterCorrectors, int nCorrectors);

    // Enable an MRF rotating zone: builds the device frame data + makes the resident flux RELATIVE (so the
    // momentum/pressure are solved in the rotating frame). Call once after construction, before stepping.
    void setMRF(
        const MRFZone& z,
        const PrimitiveMesh& m,
        const FvGeometry& g,
        const std::vector<std::string>& nonRotating = {});

    // Enable fvOptions (system/fvOptions). Empty data -> every hook below stays a no-op (bit-identical to no file),
    // exactly like fv::options::New on a case without the dict. Copies the precomputed per-cell source terms to the
    // device. Call once after construction. The momentum source feeds relaxSrc (so it reaches BOTH the predictor and
    // H()/HbyA, like the body force / OF UEqn.H fvOptions(U)); the implicit Sp lowers the momentum diagonal.
    void setFvOptions(const FvOptionsData& fo);
    // rotorDiskSource (BEM): the device rotor is built from the mesh in gpuSimpleFoam and handed over here.
    void setRotorDisk(DeviceRotorDisk r) { rotor_ = std::move(r); }
    // fvOptions scalar source for a transported scalar field (k/epsilon/...): src += Su, diag -= Sp.  No-op when absent.
    void fvOptionsAddSupScalar(
        const std::string& field,
        DeviceBuffer<scalar>* src,
        DeviceBuffer<scalar>* diag)
    {
        if (auto it = fvoScaSu_.find(field); it != fvoScaSu_.end() && src) deviceAxpy(1.0, it->second, *src);
        if (auto it = fvoScaSp_.find(field); it != fvoScaSp_.end() && diag) deviceAxpy(-1.0, it->second, *diag);
    }

    // pull device fields back to host (reconstruct for output / validation).
    std::vector<vector> U() const
    {
        const auto x=Uk_[0].host(), y=Uk_[1].host(), z=Uk_[2].host();
        std::vector<vector> u(nC_);
        for (label c=0;c<nC_;++c)
            u[c]={x[c],y[c],z[c]};
        return u;
    }
    std::vector<scalar> p()   const { return dp_.host(); }
    std::vector<scalar> k()   const { return dk_.host(); }
    std::vector<scalar> eps() const { return de_.host(); }
    std::vector<scalar> ReThetat() const { return ReThetat_.host(); }   // kOmegaSSTLM
    std::vector<scalar> gammaInt() const { return gammaInt_.host(); }
    std::vector<scalar> nut() const { return dnut_.host(); }
    // SA: the per-boundary-face nutUSpaldingWallFunction wall nut (empty for non-SA), for the viscous force.
    std::vector<scalar> nutWall() const { return dnutBndWall_.size() ? dnutBndWall_.host() : std::vector<scalar>(); }
    std::vector<scalar> cellY() const { return y_.size() ? y_.host() : std::vector<scalar>(); }   // cell wall distance (SST/SA)
    // diagnostics: the conservative face flux (internal then boundary) for continuity-error localisation.
    std::vector<scalar> phiInternal() const { return phiInt_.host(); }
    std::vector<scalar> phiBoundary() const { return phiBnd_.host(); }

private:
    const std::vector<FvPatch>& fvp_;
    DeviceSimpleControls ctl_;
    label nC_, nIf_;
    DeviceMesh dm_;
    DeviceVectorBoundary dbU_;
    DeviceBoundary dbP_, dbK_, dbEps_, dbExtrap_, dbReThetat_, dbGammaInt_;   // dbReThetat_/dbGammaInt_: kOmegaSSTLM
    DeviceWallData wall_;
    DeviceMRF mrf_;                       // optional rotating zone (inactive by default)
    // fvOptions (empty by default -> no-op). Momentum source per component (relaxSrc) + implicit Sp (diagonal);
    // scalar sources keyed by field name. See setFvOptions / the unconditional hooks in step().
    bool   hasFvoMom_ = false;
    int fvoCount_ = 0;
    DeviceBuffer<scalar> fvoMomSu_[3], fvoMomSp_;
    std::map<std::string, DeviceBuffer<scalar>> fvoScaSu_, fvoScaSp_;
    DevicePorosity por_;                  // explicitPorositySource (DarcyForchheimer), evaluated each iter
    // meanVelocityForce: accumulated pressure gradient gradP_ driving the mean velocity to Ubar (channel flow).
    bool   mvfActive_ = false;
    vector mvfFlowDir_{0,0,0};
    scalar mvfUbarMag_ = 0, mvfRelax_ = 1.0, mvfGradP_ = 0, mvfVtot_ = 0;
    DeviceBuffer<scalar> mvfMaskV_, mvfMask01_;   // V (and 1) in the zone, 0 else (= V / ones for selectionMode all)
    bool   limUActive_ = false;
    scalar limUMax_ = 0;
    DeviceBuffer<label> limUCells_;   // limitVelocity clamp
    bool   vdcActive_ = false;
    scalar vdcUMax_ = 0, vdcC_ = 1;
    DeviceBuffer<label> vdcCells_;   // velocityDampingConstraint
    // actuationDiskSource (Froude): thrust over a disk cellZone, computed each iter from the monitored upstream U.
    bool   adActive_ = false;
    vector adDiskDir_{1,0,0};
    scalar adArea_ = 0, adA_ = 0, adVtot_ = 0, adNmon_ = 0;
    DeviceBuffer<scalar> adMaskVDisk_, adMonMask01_;   // V in disk cells / 1 in monitor cells
    DeviceRotorDisk rotor_;                            // rotorDiskSource (BEM); per-cell force recomputed each iter
    DeviceBuffer<scalar> rotorFx_, rotorFy_, rotorFz_;
    AMGData amg_;
    DeviceBuffer<scalar> Uk_[3], dp_, phiInt_, phiBnd_, dk_, de_, dnut_, y_;   // y_ = cell wall distance (SST/SA)
    DeviceBuffer<scalar> hmax_;   // per-cell maxDeltaxyz (IDDES filter width); built only when ctl_.iddes
    DeviceBuffer<scalar> hwn_;    // per-cell wall-normal grid spacing (IDDES delta 3rd term); built only when ctl_.iddes
    // Transient (PIMPLE) state: ddt scheme + time steps + old-time velocity levels (U.oldTime()[.oldTime()]), rotated by
    // advanceTime(). steadyState + empty old-time buffers in the default steady SIMPLE path, where ddt is a no-op.
    DdtScheme ddtScheme_ = DdtScheme::steadyState;
    scalar    deltaT_ = 0, deltaT0_ = 0;
    DeviceBuffer<scalar> Uold_[3], Uold2_[3];
    // Turbulence old-time levels for the URANS fvm::ddt(k/eps/omega/nuTilda): dk_ (k or nuTilda) + de_ (epsilon or omega).
    DeviceBuffer<scalar> kOld_, kOld2_, e2Old_, e2Old2_;
    DeviceBuffer<scalar> ReThetatOld_, ReThetatOld2_, gammaIntOld_, gammaIntOld2_;   // kOmegaSSTLM transition old-time
    // Momentum-predictor outputs, shared with the pressure-velocity phase (PIMPLE foundation: solveMomentumPredictor()
    // fills them once per outer corrector; correctPressureVelocity() reads them). Members so both phases see them + to
    // avoid per-step reallocation. mDiagR/mUp/mLo = relaxed momentum matrix; iC/bCb = per-component boundary coeffs;
    // relaxSrc = relaxed momentum source (feeds both the predictor solve AND H()/HbyA).
    DeviceBuffer<scalar> mDiagR, mUp, mLo;
    DeviceBuffer<scalar> iC[3], bCb[3], relaxSrc[3];
    DeviceBuffer<scalar> gx, gy, gz;   // gradient workspace: grad(U) in the predictor, reused as grad(p) in the pressure phase
    DeviceBuffer<scalar> ReThetat_, gammaInt_, gammaIntEff_;                   // kOmegaSSTLM transition fields
    DeviceBuffer<scalar> dnutBndWall_;                                          // persistent Spalding wall-nut (SA warm seed)
    DeviceBuffer<scalar> nuConst_, zeroSrc_, zeroBndU_, ones_;
    DeviceBuffer<scalar> pD_, pU_, pL_, diagCp_, bp_;            // persistent pressure matrix (graph-stable addresses)
    DeviceBuffer<label>  bndIsWall_, adjustMask_;              // wall mask (true boundary nut); adjustPhi adjustable mask
    DeviceBuffer<scalar> bndY_, nuBndConst_;                    // nearWallDist y per boundary face; nu over bnd faces
    bool   hasMixed_ = false;                                  // any freestreamVelocity/Pressure (mixed) patch present
    bool   hasPiov_ = false;                                   // any pressureInletOutletVelocity (directionMixed) patch present
    bool   hasSym_ = false;                                    // any slip/symmetry patch present (general normal)
    bool   hasTotalP_ = false;                                 // any totalPressure p patch present (per-step refValue)
    bool   hasCyclic_ = false;                                 // any cyclic (periodic) interface -> Jacobi-PCG pressure (no AMG)
    DeviceCyclic cyc_;                                          // periodic interface coupling (OF updateInterfaceMatrix)
    bool   hasAMI_ = false;                                     // any cyclicAMI interface -> Jacobi-PCG pressure (no AMG)
    DeviceAMI    ami_;                                          // cyclicAMI weighted-stencil coupling (translational path)
};

} // namespace brae
