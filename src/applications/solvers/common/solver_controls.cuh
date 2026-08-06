#pragma once
// SIMPLE solver configuration + per-step residual report -- the PODs that parameterise DeviceSimpleSolver.
// Split out of device_simple_foam.cuh so the solver interface, PIMPLE, rhoSimple etc. can share the controls
// without pulling in the full solver implementation. Pure data (+ the model-coeff PODs); no device code.
#include "cf_types.cuh"
#include "kepsilon_coeffs.cuh"
#include "komega_sst_coeffs.cuh"
#include "spalart_coeffs.cuh"
#include "smagorinsky_coeffs.cuh"
#include <string>

namespace brae {

// Which nut wall function to apply, chosen from the 0/nut boundaryField TYPE (matching OpenFOAM),
// NOT from the turbulence model. nutk = k-based stepwise log law (nutkWallFunction); Spalding =
// velocity-based Spalding blend (nutUSpaldingWallFunction); Blended = velocity-based binomial n=4
// blend (nutUBlendedWallFunction). All three are honoured on any RAS model, exactly as OF does.
enum class NutWall { Nutk, Spalding, Blended, NutU };   // NutU = nutUWallFunction (STEPWISE blender)

struct DeviceSimpleControls
{
    scalar nu = 1e-5, relaxU = 0.7, relaxP = 0.3, relaxK = 0.7, relaxEps = 0.7;
    vector bodyForce{0, 0, 0};                     // constant momentum source (drives periodic/cyclic channels). +V*g.
    scalar tolU = 1e-8, tolP = 1e-7, tolKE = 1e-8;
    scalar relTolU = 0.0, relTolP = 0.0, relTolKE = 0.0;   // solver relTol (fvSolution solvers.{U,p,k,epsilon}.relTol). 0 = abs tol.
    int    bicgCheckEvery = 1;      // batched convergence for ALL BiCGStab solves (momentum + k/eps); BRAE_BICG_CHECK_EVERY.
    bool   turbulent = false;
    bool   useGraph  = true;     // replay the pressure V-cycle from a cached CUDA graph (#7c-loop)
    bool   bounded   = false;    // "bounded Gauss upwind": add -Sp(div(phi),U). Set from fvSchemes; default off.
    bool   consistent = false;   // SIMPLEC (rAtU consistency correction, p=1). Set from fvSolution SIMPLE.consistent.
    bool   linearUpwind = false; // div(phi,U) "linearUpwind": add the deferred gradient correction. Set from fvSchemes.
    bool   linearUpwindV = false;// div(phi,U) "linearUpwindV": linearUpwind + OF's vector direction limiter (also sets linearUpwind).
    bool   lust = false;         // div(phi,U) "LUST": deferred correction = 0.75*linear + 0.25*linearUpwind (OF LUST.H).
    bool   nonOrth = false;      // laplacian "corrected"|"limited": nonOrthDeltaCoeffs implicit + explicit corrVec.grad correction. Set from fvSchemes.
    scalar nonOrthLimit = 1.0;   // snGrad "limited <psi>" coeff (OF fv::limitedSnGrad); 1.0 = "corrected" (unlimited). Set from fvSchemes.
    int    nNonOrth = 0;         // SIMPLE.nNonOrthogonalCorrectors: extra pressure-correction passes (pEqn re-solved nNonOrth+1 times). Set from fvSolution.
    scalar gradULimitK = 0.0;    // grad(U) "cellLimited Gauss linear <k>" coeff (OF cellLimitedGrad<minmod>); 0 = unlimited. Set from fvSchemes.
    bool   limitedK = false, limitedEps = false;  // div(phi,k|epsilon) "limitedLinear": implicit limited weight. Set from fvSchemes.
    bool   luK = false, luEps = false;             // div(phi,k|epsilon|nuTilda) "linearUpwind": deferred gradient correction. Set from fvSchemes.
    bool   gsK = false, gsEps = false;             // scalar linear solver = smoothSolver+symGaussSeidel (read from fvSolution solvers.{k|nuTilda} / {epsilon|omega}).
    bool   gsU = false;                            // momentum linear solver = smoothSolver+(sym)GaussSeidel (read from fvSolution solvers.U).
    scalar twoBykK = 2.0, twoBykEps = 2.0;         // 2/max(k_,SMALL) from the limitedLinear coefficient (k_=1 -> 2).
    // div(phi,h|e) and div(phi,K|Ekp) -- the ENERGY equation's convection scheme (rhoSimpleFoam). Read from
    // fvSchemes like every other div scheme; brae used to hardcode upwind here and silently ignore what the
    // case asked for, which is a first-order downgrade that still converges. The K/Ekp term takes the same
    // scheme because OF's tutorials point div(phi,K) at the same entry ($energy) as div(phi,e).
    // div(phi,h|e) "bounded": OF's boundedConvectionScheme subtracts fvm::Sp(fvc::surfaceIntegrate(phi), he).
    // It vanishes at converged continuity, so every steady duct gate passes without it -- but it is a real
    // STABILISER while div(phi) != 0, and the NACA0012 case at Mach 0.72 diverges in one iteration without it.
    bool   boundedHe = false;
    // Same story for the turbulence scalars. OF's `bounded` is a per-SCHEME wrapper
    // (boundedConvectionScheme wraps one div), so div(phi,k) can be bounded while div(phi,U) is not.
    // brae read `bounded` only off the div(phi,U) line and reused that one flag for every scalar, so
    // demo/delta and demo/f16 (`div(phi,U) Gauss LUST` + `div(phi,nuTilda) bounded ...`) ran nuTilda
    // UNBOUNDED. The term is -Sp(div(phi),f): on an upwind row it takes diag from sum|outflow phi| to
    // sum|inflow phi| = sum|offdiag|, i.e. it guarantees marginal diagonal dominance. Without it any
    // cell with net inflow (everywhere, before continuity converges) is not diagonally dominant.
    bool   boundedK   = false;   // div(phi,k) / div(phi,nuTilda)
    bool   boundedEps = false;   // div(phi,epsilon) / div(phi,omega)
    bool   limitedHe = false;                      // div(phi,h|e) "limitedLinear"
    bool   luHe = false;                           // div(phi,h|e) "linearUpwind"
    scalar twoBykHe = 2.0;
    // div(phi,K|Ekp), the KINETIC term -- a separate fvSchemes entry from div(phi,h|e) and a separate
    // discretisation in OF. All four keys used to OR into the He slots, so a case asking for
    // `div(phi,e) linearUpwind` + `div(phi,K) upwind` ran the kinetic term second-order anyway (and the
    // reverse ran the energy equation first-order). They agree in every stock tutorial, where both are
    // `$energy` -- which is exactly why the merge went unnoticed.
    bool   boundedKin = false;
    bool   limitedKin = false;
    bool   luKin = false;
    scalar twoBykKin = 2.0;
    bool   foundKinScheme = false;                 // false -> mirror the He slots (the old behaviour)
    // OF's RAS/LES `turbulence` switch (RASModel.C:70, getOrDefault<Switch>("turbulence", true)). `off`
    // makes correct() return immediately: k, epsilon/omega and nut stay FROZEN at their initial values
    // while the momentum equation keeps using that frozen nut. It is NOT the same as `simulationType
    // laminar`, where nut is zero and never read. brae never read the switch, so a case asking to freeze
    // the turbulence ran a fully live model. Found by dict_audit (E5) once the audit was made to run on
    // REFUSED cases -- aerofoilNACA0012 sets it, and aerofoilNACA0012 refuses on fvOptions.
    bool   turbulenceOn = true;
    // B1: SIMPLE/transonic. OF's rhoSimpleFoam pEqn.H takes a different branch entirely -- the pressure
    // equation gains an IMPLICIT convection term fvm::div(phid, p) with phid = (psi_f/rho_f)*phiHbyA, the
    // part of phiHbyA now carried implicitly is subtracted off, and the matrix is relaxed (the subsonic
    // branch relaxes only the FIELD, never the equation). Running a transonic case down the subsonic
    // branch converges to a wrong answer in silence, which is why this was a refusal until now.
    bool   transonic = false;
    // OF fvMatrix::relax() applies relaxationFactors/equations/<field> and does NOTHING when the entry is
    // absent (fvMatrix.C:1250-1263 -- it only relaxes if relaxEquation() finds one). Both facts matter:
    // the factor AND whether it was given.
    bool   hasRelaxPEqn = false;
    scalar relaxPEqn = 1.0;
    // grad(<turbulence scalar>) / grad(energy) cellLimited coefficients. Separate from gradULimitK: OF takes
    // one gradScheme entry per field, and aerofoilNACA0012 limits grad(U), grad(k) and grad(omega) alike.
    scalar gradKLimitK = 0.0;                      // grad(k) / grad(omega) / grad(epsilon) / grad(nuTilda)
    scalar gradHeLimitK = 0.0;                     // grad(h) / grad(e)
    // The cellLimited coefficient of the gradient the KINETIC term's linearUpwind uses. Separate from
    // gradHeLimitK because fvSchemes gives div(phi,K|Ekp) its own entry, which may name a different
    // gradient from div(phi,h|e); it falls back to the energy's when the case omits it.
    scalar gradKinLimitK = 0.0;                    // grad(K) / grad(Ekp)
    KEpsilonCoeffs keCoeffs;                       // k-eps model coeffs (default = OF); read from turbulenceProperties RAS.kEpsilonCoeffs.
    bool   sst = false;                            // RASModel kOmegaSST (the "second turbulence scalar" eps slot holds omega).
    bool   lm = false;                             // RASModel kOmegaSSTLM (sst + Langtry-Menter gamma-ReThetat transition).
    KOmegaSSTCoeffs ksstCoeffs;                    // kOmegaSST coeffs (default = OF); read from RAS.kOmegaSSTCoeffs.
    bool   sa = false;                             // RASModel SpalartAllmaras (one-equation: the "k" slot holds nuTilda; no 2nd scalar).
    bool   des = false;                            // SpalartAllmarasDDES/kOmegaSSTDDES (simulationType LES): DES length-scale limiter; needs sa/sst=true.
    bool   iddes = false;                          // SpalartAllmarasIDDES: the improved (WMLES-capable) length scale on the SA DES path; implies des+sa.
    SpalartAllmarasCoeffs saCoeffs;                // SA coeffs (default = OF) + nutUSpaldingWallFunction E/kappa + CDES (DES) + IDDES blending constants.
    bool   les = false;                            // pure LES Smagorinsky (simulationType LES): ALGEBRAIC sub-grid nut,
                                                   // NO transport scalar (no k/epsilon/omega/nuTilda). Mutually exclusive with sa/sst/des.
    SmagorinskyCoeffs smagCoeffs;                  // Smagorinsky coeffs (default = OF Ck=0.094, Ce=1.048); read from LES.SmagorinskyCoeffs.
    NutWall nutWall = NutWall::Nutk;               // nut wall function, read from the 0/nut wall BC TYPE (not the model),
                                                   // so nutUBlended/nutUSpalding are honoured on kEpsilon/kOmegaSST like OF.
    scalar atmZ0 = 0.0;                            // atmNutkWallFunction roughness length z0 (>0 -> rough-wall nut on the
    bool   atmBoundNut = true;                     // k-based path); read from the 0/nut wall BC. 0 -> smooth nutkWallFunction.
    bool   needRef = false;                        // no fixedValue-p patch -> singular pressure: adjustPhi + setReference.
    label  pRefCell = 0;
    scalar pRefValue = 0.0;                        // pressure reference (fvSolution SIMPLE.pRefCell/pRefValue).
    int    pcgCheckEvery = 1;      // pressure AMG-PCG residual-read cadence (BRAE_PCG_CHECK_EVERY). 1 = exact per-iter
                                   // (bit-identical); K>1 cuts (K-1)/K of the PCG host syncs (overshoots conv by <K).
    bool   corrScaling = false;    // AMG coarse-correction scaling + flexible CG (BRAE_CORR_SCALING). Cuts AMG cycles
                                   // ~2x at scale (graded meshes); nonlinear precond -> not bit-identical to off.
    std::string caseDir = ".";     // case directory, for the AMG hierarchy cache (constant/polyMesh/.brae_amgcache).
    bool   writeCache = false;     // write the mesh + AMG caches this run (set by -partition or BRAE_MESH_CACHE).
};

// OF-style per-field solve report for one SIMPLE step (matches OpenFOAM's "Solving for Ux/Uy/Uz/p" + continuity block).
struct DeviceSimpleResidual
{
    scalar Ux = 0, Uy = 0, Uz = 0, p = 0, pFinal = 0;
    scalar UxFinal = 0, UyFinal = 0, UzFinal = 0;
    int UxIters = 0, UyIters = 0, UzIters = 0;     // U per-component final + nIter
    int pIters = 0;
    scalar contLocal = 0, contGlobal = 0;          // time step continuity errors, raw (driver applies deltaT + cumulative)
};

} // namespace brae
