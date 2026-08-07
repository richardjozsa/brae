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
#include "thermo_model.cuh"   // hConstHeToT: boundary he -> boundary T for the written field
#include "device_pcg.cuh"
#include "device_simple.cuh"
#include "device_boundary.cuh"
#include "thermo_types.cuh"
#include "device_thermo.cuh"
#include "device_energy.cuh"
#include "rho_simple_controls.cuh"
#include "device_cyclic.cuh"
#include "device_ami.cuh"
#include "device_interface.cuh"   // interface<Op>() overloads dispatching to the cyclic/AMI backends
#include "device_kepsilon.cuh"
#include "device_komega_sst.cuh"   // deviceS2/deviceF2/deviceNutSST for the startup validate() correctNut
#include "device_smagorinsky.cuh"  // deviceSmagorinskyNut for the pure-LES (algebraic sub-grid nut) path
#include "device_coded_bc.cuh"     // NVRTC device-resident coded (runtime-compiled) boundary conditions
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
    // E7: re-run OF's validate() ONCE the thermo exists, which is where OF does it.
    //
    // OF constructs the thermo, THEN the turbulence model, THEN calls validate() -> correctNut(). brae
    // constructs the solver (validate included) before the compressible driver has seeded the thermo, so
    // its startup correctNut saw a boundary velocity built from a PLACEHOLDER density: a
    // flowRateInletVelocity patch falls back to `rhoInlet` until a real rho exists, and on rhoTI that is
    // 1.0 against a true 1.161, giving an inlet U of 58.85 m/s instead of 50.69 against a uniform internal
    // 50. The resulting du/dx = 354 makes sqrt(S2) = 500.6 and trips the SST Bradshaw limiter, so
    // nut/(k/omega) came out 0.2477 where OF has 1 -- predicted to six decimals by that arithmetic before
    // any code changed. Invisible with `turbulence on` (the next iteration overwrites the seed), and the
    // answer with `turbulence off`.
    void revalidateAfterThermo();
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
    void setDdtScheme(DdtScheme s, scalar ocCoeff = scalar(1))
    {
        ddtScheme_ = s;
        ocCoeff_   = ocCoeff;
        if (s == DdtScheme::CrankNicolson)   // allocate + zero each transported variable's stored old ddt (ddt0)
        {
            const std::vector<scalar> z(nC_, scalar(0));
            for (int k = 0; k < 3; ++k) Uddt0_[k].copyFrom(z);
            if (ctl_.turbulent && !ctl_.les)
            {
                kddt0_.copyFrom(z);
                if (!ctl_.sa) e2ddt0_.copyFrom(z);
                if (ctl_.lm) { ReThetatddt0_.copyFrom(z); gammaIntddt0_.copyFrom(z); }
            }
        }
    }
    // Advance one time level: store U.oldTime()[.oldTime()] + roll deltaT -> deltaT0 (OF runTime++ / GeometricField::oldTime).
    void advanceTime(scalar deltaT);
    // One PIMPLE time step: advanceTime, then nOuterCorrectors x { momentum predictor (ddt folded in); nCorrectors x
    // pressure-velocity correction; turbulence }, then continuity errors. Returns the outer-loop residual signal.
    DeviceSimpleResidual pimpleStep(scalar deltaT, int nOuterCorrectors, int nCorrectors);

    // Steady COMPRESSIBLE step (rhoSimpleFoam). Composes the same three phases as simpleStep/pimpleStep
    // with the energy equation and the thermo update inserted, in OF's rhoSimpleFoam.C order:
    //
    //     UEqn -> EEqn -> pEqn -> thermo.correct() -> rho.relax() -> turbulence
    //
    // EEqn sits between momentum and pressure because the pressure equation needs the density the NEW
    // temperature implies, not the previous one. Requires setCompressible() first.
    DeviceSimpleResidual rhoSimpleStep();

    // The compressible state, so the driver can seed T/he before the loop and read T back after it.
    // The solver owns the buffers (they are sized against its mesh); the driver owns what goes in them.
    DeviceThermo& thermo() { return th_; }
    const DeviceThermo& thermo() const { return th_; }

    // Switch the solver into compressible mode. dbHe is the enthalpy boundary (built from the case's 0/T
    // via deviceEnergyBoundaryFromT); th is allocated and seeded by the caller so the driver owns field
    // construction, exactly as it already owns U/p.
    // Per-boundary-face Prt from 0/alphat: alphatWallFunction patches carry their own (OF default 0.85),
    // every other face uses the turbulence model's (OF default 1.0). Optional -- absent, the model's is used.
    void setAlphatPrt(const std::vector<scalar>& prtFace);
    // Energy linear solver from fvSolution solvers.(h|e). Was hardcoded tol=1e-10, relTol=0, BiCGStab.
    void setEnergySolver(scalar tol, scalar relTol, bool useGS)
    { eTol_ = tol; eRelTol_ = relTol; eUseGS_ = useGS; }
    void setCompressible(
        const ThermoCoeffs& tc,
        const RhoSimpleControls& rc,
        DeviceBoundary dbHe);

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
    // Device-side p, for the compressible thermo update (which must not round-trip through the host).
    const DeviceBuffer<scalar>& pDevice() const { return dp_; }
    // Turbulence magnitude for the divergence tripwire: sum|k| + sum|second scalar|, as a DEVICE reduction
    // (no D2H of the fields). Used only to detect blow-up, never as a convergence measure.
    scalar turbSumMag() const
    {
        scalar s = 0;
        if (dk_.size()) s += deviceSumMag(dk_);
        if (de_.size()) s += deviceSumMag(de_);
        return s;
    }
    // Per-face turbulent-inlet data, so the boundary values can be refreshed per iteration (OF parity).
    void setTurbulentInlets(const std::vector<label>& tiMask, const std::vector<scalar>& tiIntensity,
                            const std::vector<label>& mlMask, const std::vector<scalar>& mlLength)
    {
        bool any = false;
        for (label m : tiMask) if (m) { any = true; break; }
        if (!any) for (label m : mlMask) if (m) { any = true; break; }
        if (!any) return;
        tiMask_.copyFrom(tiMask); tiIntensity_.copyFrom(tiIntensity);
        mlMask_.copyFrom(mlMask); mlLength_.copyFrom(mlLength);
        hasTurbInlet_ = true;
    }
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
    // The SOLVED boundary values, so the written boundaryField reports what the solve used rather than
    // echoing the input file. nut is the one that matters most: on a wall its value comes from the wall
    // function, and post-processing (yPlus, wall shear, force split) reads it straight back.
    std::vector<scalar> nutBoundary() const
    {
        return nutBndAll_.size() ? nutBndAll_.host() : std::vector<scalar>();
    }
    // Generic: evaluate any cell field on its own boundary descriptor.
    std::vector<scalar> boundaryOf(const DeviceBoundary& db, const DeviceBuffer<scalar>& f) const
    {
        if (!db.n || !f.size()) return {};
        DeviceBuffer<scalar> b;
        deviceBCValue(db, f, b);
        return b.host();
    }
    // T on the boundary. he is the solved variable, so the boundary T is T(he_b) -- evaluating he's own
    // BC (which now resolves inletOutlet per iteration) and converting, rather than echoing 0/T.
    std::vector<scalar> TBoundary() const
    {
        if (!compressible_ || !dbHe_.n || !th_.he.size()) return {};
        DeviceBuffer<scalar> heB;
        deviceBCValue(dbHe_, th_.he, heB);
        std::vector<scalar> h = heB.host(), t(h.size());
        for (std::size_t i = 0; i < h.size(); ++i) t[i] = hConstHeToT(h[i], tc_);
        return t;
    }
    // U on the boundary: each component evaluated against its own descriptor, then recombined. Needed
    // because a vector BC can be per-component (slip/symmetry set vf = |n_k| per component), so there is
    // no single scalar evaluation that covers it.
    std::vector<vector> UBoundary() const
    {
        if (!dbU_.comp[0].n) return {};
        DeviceBuffer<scalar> b0, b1, b2;
        deviceBCValue(dbU_.comp[0], Uk_[0], b0);
        deviceBCValue(dbU_.comp[1], Uk_[1], b1);
        deviceBCValue(dbU_.comp[2], Uk_[2], b2);
        const std::vector<scalar> h0 = b0.host(), h1 = b1.host(), h2 = b2.host();
        std::vector<vector> v(h0.size());
        for (std::size_t i = 0; i < h0.size(); ++i) v[i] = vector{h0[i], h1[i], h2[i]};
        return v;
    }
    // nuTilda shares the k slot on the SA path.
    std::vector<scalar> nuTildaBoundary() const { return boundaryOf(dbK_, dk_); }
    std::vector<scalar> kBoundary()   const { return boundaryOf(dbK_,   dk_); }
    std::vector<scalar> epsBoundary() const { return boundaryOf(dbEps_, de_); }
    std::vector<scalar> pBoundary()   const { return boundaryOf(dbP_,   dp_); }
    std::vector<scalar> phiBoundary() const { return phiBnd_.host(); }
    // rho on the boundary -- the EOS result the solve actually used, already maintained per face by the
    // pressure equation (phiHbyA is rho-weighted with it). Not derivable from the T/p descriptors here,
    // which is why it needs its own accessor rather than boundaryOf().
    std::vector<scalar> rhoBoundary() const { return compressible_ ? rhoBnd_.host() : std::vector<scalar>{}; }

    // CrankNicolson ddt0 (stored old ddt) persistence -- get/set for seamless restart (the driver writes+reads these).
    bool isCrankNicolson() const { return ddtScheme_ == DdtScheme::CrankNicolson; }
    bool cnWarm() const { return cnWarm_; }
    void setCnWarm(bool w) { cnWarm_ = w; }
    // Prime the CrankNicolson time state on restart: deltaT_=deltaT0 makes the FIRST advanceTime set deltaT0_=deltaT0
    // (so the first post-restart step is full CN, not the Euler bootstrap), and the ddt0 recurrence is already warm.
    void setCnRestart(scalar deltaT0) { deltaT_ = deltaT0; cnWarm_ = true; }

    // NVRTC device-coded BC (codedFixedValue on U): compile `code` once into a device kernel over the patch's boundary
    // faces [offset, offset+count) in the flat boundary order. Applied each step in solveMomentumPredictor (writes the
    // patch's dbU_ refValue on the device). setTime feeds the current time to the snippet's `t`.
    // target: 0=U (vector), 1=p, 2=k/nuTilda, 3=second (omega/epsilon). mixed=true -> codedMixed (also writes valueFraction).
    void addCodedBC(const std::string& name, const std::string& code, int offset, int count, int target = 0, bool mixed = false);
    void setTime(scalar t) { time_ = t; }
    std::vector<vector> Uddt0() const
    {
        const auto x = Uddt0_[0].host(), y = Uddt0_[1].host(), z = Uddt0_[2].host();
        std::vector<vector> r(x.size());
        for (std::size_t i = 0; i < x.size(); ++i) r[i] = vector{x[i], y[i], z[i]};
        return r;
    }
    void setUddt0(const std::vector<vector>& v)
    {
        std::vector<scalar> x(v.size()), y(v.size()), z(v.size());
        for (std::size_t i = 0; i < v.size(); ++i) { x[i] = v[i].x; y[i] = v[i].y; z[i] = v[i].z; }
        Uddt0_[0].copyFrom(x); Uddt0_[1].copyFrom(y); Uddt0_[2].copyFrom(z);
    }
    std::vector<scalar> kddt0()  const { return kddt0_.host(); }
    void setKddt0(const std::vector<scalar>& v)  { kddt0_.copyFrom(v); }
    std::vector<scalar> e2ddt0() const { return e2ddt0_.host(); }
    void setE2ddt0(const std::vector<scalar>& v) { e2ddt0_.copyFrom(v); }
    std::vector<scalar> reThetatddt0() const { return ReThetatddt0_.host(); }
    void setReThetatddt0(const std::vector<scalar>& v) { ReThetatddt0_.copyFrom(v); }
    std::vector<scalar> gammaIntddt0() const { return gammaIntddt0_.host(); }
    void setGammaIntddt0(const std::vector<scalar>& v) { gammaIntddt0_.copyFrom(v); }

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
    // limitTemperature. Held as HE limits, not T limits: OF converts once with the case thermo and clamps
    // the energy variable the equation actually solved (limitTemperature::correct(he)).
    bool   limTActive_ = false;
    bool   limTAllCells_ = false;     // selectionMode all -> the he BOUNDARY is clamped too
    scalar limTMin_ = 0, limTMax_ = 0;
    DeviceBuffer<label> limTCells_;
    // fixedTemperatureConstraint: OF constrains the ENERGY matrix (eqn.setValues), so this is carried as a
    // per-cell mask + per-cell he value and handed to the same setValues path the eps wall function uses.
    bool                fixTActive_ = false;
    DeviceWallData      fixTMask_;    // only isWallCell is used by the setValues kernels
    DeviceBuffer<scalar> fixTHe_;
    // scalarFixedValueConstraint on k / epsilon|omega. Per-cell mask + per-cell value, handed to the same
    // setValues path; OF applies it before the eps wall function (kEpsilon.C:266 then :267).
    bool                 fixScaK_ = false, fixScaE_ = false;
    DeviceBuffer<label>  fixScaMask_;
    DeviceBuffer<scalar> fixScaKVal_, fixScaEVal_;
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
    scalar    ocCoeff_ = 1;         // CrankNicolson off-centring (fvSchemes "CrankNicolson <oc>"); 0=Euler-like, 1=pure CN
    bool      cnWarm_  = false;     // CrankNicolson: the ddt0 recurrence has done its first (Euler-startup) update
    // NVRTC device-coded BCs (codedFixedValue). bndCf* = flat boundary face centres (patch order, cyclic excluded).
    struct SolverCodedBC { CodedBcKernel kernel; int offset = 0, count = 0; int target = 0; bool mixed = false; };   // target: 0=U 1=p 2=k 3=second
    std::vector<SolverCodedBC> codedBCs_;
    DeviceBuffer<scalar> bndCfX_, bndCfY_, bndCfZ_;
    scalar    time_ = 0;            // current time, fed to the coded-BC snippet's `t`
    DeviceBuffer<scalar> Uold_[3], Uold2_[3];
    // Turbulence old-time levels for the URANS fvm::ddt(k/eps/omega/nuTilda): dk_ (k or nuTilda) + de_ (epsilon or omega).
    DeviceBuffer<scalar> kOld_, kOld2_, e2Old_, e2Old2_;
    DeviceBuffer<scalar> ReThetatOld_, ReThetatOld2_, gammaIntOld_, gammaIntOld2_;   // kOmegaSSTLM transition old-time
    DeviceBuffer<scalar> Uddt0_[3], kddt0_, e2ddt0_, ReThetatddt0_, gammaIntddt0_;   // CrankNicolson stored old ddt (per variable); allocated only for CN
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

    // Compressible state (rhoSimpleFoam). compressible_ stays false for every incompressible case, so
    // none of this is reachable from the existing solvers and the steady/PIMPLE paths are untouched.
    bool compressible_ = false;
    ThermoCoeffs tc_{};
    RhoSimpleControls rc_{};
    DeviceThermo th_{};
    DeviceBoundary dbHe_{};
    DeviceBuffer<scalar> pD_, pU_, pL_, diagCp_, bp_;            // persistent pressure matrix (graph-stable addresses)
    DeviceBuffer<label>  bndIsWall_, adjustMask_;              // wall mask (true boundary nut); adjustPhi adjustable mask
    // kEpsilon: 1 on boundary faces whose 0/nut BC is 'calculated'. OF fills those by field assignment
    // (Cmu*k_b^2/eps_b) rather than extrapolating the cell, and the two differ by >12x at a fixed-k inlet.
    DeviceBuffer<label>  nutCalcMask_;
    bool hasNutCalc_ = false;
    DeviceBuffer<scalar> bndY_, nuBndConst_;                    // nearWallDist y per boundary face; nu over bnd faces
    // Compressible only: per-boundary-face rho and laminar mu, refreshed each outer iteration from the
    // boundary p/he. OF gets these from rho.boundaryField()[patchi] and transport_.mu(patchi); they feed
    // muEff_b = mu_b + rho_b*nut_b and the kinematic nu_b = mu_b/rho_b the wall functions ask for.
    // SIMPLEC: (rAtU - rAU) for the HbyA correction, rho*(rAtU - rAU) for the FLUX correction on the
    // compressible path (OF pcEqn.H). Identical buffers when incompressible.
    DeviceBuffer<scalar> drAtUFlux_;
    scalar eTol_ = 1e-10, eRelTol_ = 0.0;
    bool   eUseGS_ = false;
    DeviceBuffer<scalar> rhoBnd_, muBnd_, nuWallBnd_;
    // The BOUNDARY half of the solver's own `rho` field -- the one OF relaxes. rhoBnd_ above is
    // thermo.rho() at the boundary, recomputed fresh from p_b and T_b every time it is asked for, which
    // is right for everything that reads thermo.rho() (nu(patchi), alphaEff(patchi), the turbulence
    // model). It is WRONG for `phiHbyA = fvc::interpolate(rho)*fvc::flux(HbyA)`, which reads the solver's
    // rho field, and that field is assigned once per outer iteration and then relaxed.
    DeviceBuffer<scalar> rhoBndP_, rhoBndPPrev_;
    // pEqn.relax() source (D - D0)*p. Absolute, so it is added to bp_ AFTER deviceFoldPressure (which
    // multiplies divPhi by the cell volume); a member so the fold site can see it.
    DeviceBuffer<scalar> pRelaxSrc_;
    // flowRateInletVelocity, massFlowRate form: OF recomputes avgU = -mdot/gSum(rho*magSf) every call, so
    // it moves with the solution. frMagSf_ is magSf masked to the flowRate patches (0 elsewhere), making
    // the patch sum a single dot product against the live boundary rho; frN_ holds the outward normals.
    struct FlowRatePatch { scalar mdot; };
    std::vector<FlowRatePatch> frPatches_;
    std::vector<DeviceBuffer<scalar>> frMagSf_;
    DeviceBuffer<scalar> frNx_, frNy_, frNz_;
    // turbulentIntensityKineticEnergyInlet / turbulentMixingLength*Inlet: per-face masks + coefficients so
    // the inlet k and epsilon|omega can be REBUILT each outer iteration, as OF's updateCoeffs does. Frozen
    // set-up values are exact only when the inlet U never moves; flowRateInletVelocity moves it.
    DeviceBuffer<label>  tiMask_;      // 1 where k has a turbulentIntensityKineticEnergyInlet face
    DeviceBuffer<scalar> tiIntensity_;
    DeviceBuffer<label>  mlMask_;      // 1 = epsilon, 2 = omega
    DeviceBuffer<scalar> mlLength_;
    bool hasTurbInlet_ = false;
    bool hasFlowRate_ = false;
    DeviceBuffer<label>  wfBndIdx_;   // wall-face -> boundary-face index (built once)
    DeviceBuffer<scalar> wfNu_;       // nu = mu_b/rho_b gathered onto wall faces, for omegaWallFunction/G0
    DeviceBuffer<scalar> nutBndAll_;  // nut at boundary faces (wall-function value on walls), for alphat_b
    DeviceBuffer<scalar> prtBnd_;     // per-face Prt: the alphatWallFunction's on its patches, the model's elsewhere
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
