// brae_rhoSimpleFoam -- steady COMPRESSIBLE SIMPLE, single-GPU device-resident.
//
// Reads a standard OpenFOAM rhoSimpleFoam case and runs the whole loop on the GPU via
// DeviceSimpleSolver::rhoSimpleStep -- the same three composable phases the steady and PIMPLE solvers
// use, with the energy equation and the thermo update inserted:
//
//     UEqn -> EEqn -> pEqn -> thermo.correct() -> rho.relax() -> turbulence
//
// Scope today is SUBSONIC, perfectGas + hConst + (sutherland | const), laminar. Anything outside that is
// refused at start-up rather than run with the wrong physics -- see readThermoCoeffs, which rejects an
// unsupported thermoType by name, and the transonic check below.
//
//   brae_rhoSimpleFoam -case <caseDir>
//
// Note on p: this solver reads ABSOLUTE pressure in Pa, not the kinematic p/rho the incompressible
// solvers use. A case copied from simpleFoam with p in m2/s2 will run and give nonsense, so the
// dimensions line is checked.

#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "foam_field_writer.cuh"
#include "foam_dict.cuh"
#include "fvc.cuh"
#include "device_simple_foam.cuh"
#include "thermo_parse.cuh"
#include "rho_simple_controls.cuh"
#include "turbulence_setup.cuh"   // readTurbulenceModel + readTurbulenceFields (shared with simpleFoam/pimpleFoam)
#include "patch_entry_lookup.cuh"   // findPatchEntry: OF patch/group/regex resolution
#include "brae_notice.cuh"   // noticeIgnored/Approximated/Defaulted: never drop an input silently
#include "start_time.cuh"   // OF startFrom latestTime/firstTime (shared with simpleFoam)
#include "residual_control.cuh"   // OF simpleControl::criteriaSatisfied (shared with simpleFoam)
#include "dict_audit.cuh"   // name every dict entry brae read and then ignored
#include "write_control.cuh"   // OF writeControl/writeInterval/purgeWrite cadence (shared with simpleFoam)
#include "scheme_parse.cuh"        // parseFvSchemesControls
#include "linear_solver_setup.cuh" // readLinearSolverControls + readEnergySolverControls (shared with gpuSimpleFoam)
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <filesystem>
#include <cmath>
#include <algorithm>

using namespace brae;

int main(int argc, char** argv)
{
    try
    {
        std::string caseDir = ".";
        for (int i = 1; i < argc; ++i)
        {
            const std::string a = argv[i];
            if (a == "-case" && i + 1 < argc) caseDir = argv[++i];
        }

        const FoamDict controlDict = readDict(caseDir + "/system/controlDict");
        const FoamDict fvSolution = readDict(caseDir + "/system/fvSolution");
        // Read here rather than at its point of use so the audit guard below can outlive it -- the guard
        // holds pointers and must be destroyed BEFORE the dicts it reports on.
        const FoamDict turbProps = readDict(caseDir + "/constant/turbulenceProperties");

        // E5: report on EVERY exit, refusals included. Declared after all three dicts so it is destroyed
        // first. The two stock tutorials most worth auditing (aerofoilNACA0012, angledDuct) both refuse on
        // fvOptions, so before this they were the only cases never audited.
        DictAuditScope audit;
        audit.add(controlDict, "system/controlDict");
        audit.add(fvSolution, "system/fvSolution");
        audit.add(turbProps, "constant/turbulenceProperties");

        const ThermoCoeffs tc = readThermoCoeffs(caseDir, &fvSolution);   // share the dict so dict_audit sees these lookups
        const RhoSimpleControls rc = readRhoSimpleControls(fvSolution, tc.internalEnergy);

        PrimitiveMesh m;
        m.read(caseDir + "/constant/polyMesh");
        FvGeometry g;
        g.build(m);
        const std::vector<FvPatch> fvp = buildPatches(m, g);
        const label nC = m.nCells();

        // C6: startFrom was ignored here -- the start directory was hardcoded to "0", so `startFrom
        // latestTime` (the standard way to CONTINUE a compressible run) silently restarted from scratch
        // and then converged to a perfectly good answer, having discarded the restart.
        const std::string t0 = caseDir + "/" + resolveStartTime(
            caseDir,
            controlDict.wordOr("startFrom", "startTime"),
            controlDict.wordOr("startTime", "0"));
        GeometricField<vector> U = buildField<vector>(readField<vector>(t0 + "/U"), fvp, nC);
        GeometricField<scalar> p = buildField<scalar>(readField<scalar>(t0 + "/p"), fvp, nC);
        GeometricField<scalar> T = buildField<scalar>(readField<scalar>(t0 + "/T"), fvp, nC);
        U.evaluateBoundary();
        p.evaluateBoundary();
        T.evaluateBoundary();

        // Initial density, so the starting flux is a MASS flux like every later one. Without this the
        // first pressure equation sees a volumetric phiHbyA and the first iteration is inconsistent.
        std::vector<scalar> rho0(nC);
        for (label i = 0; i < nC; ++i) rho0[i] = p.internal[i] / (tc.R * T.internal[i]);
        const SurfaceScalarField phi = fvc::rhoFlux(rho0, U, m, g, fvp);

        // Resolve pMaxFactor/pMinFactor into absolute limits the way OF does: the reference is taken from
        // the p BOUNDARY values (pressureControl.C scans p.boundaryField() over patches that fix a value).
        // Without a fixed-pressure patch OF errors out and tells the user to give pMax/pMin directly, so
        // brae does the same rather than inventing a reference from the internal field.
        {
            RhoSimpleControls& rcm = const_cast<RhoSimpleControls&>(rc);
            if (rcm.pMaxFactor > 0.0 || rcm.pMinFactor > 0.0)
            {
                // OF scans ONLY the patches that FIX a value (pressureControl.C:75-77:
                // `if (pbf[patchi].fixesValue())`), because pMaxFactor/pMinFactor scale a KNOWN reference
                // pressure -- a zeroGradient or calculated patch carries whatever the field currently
                // holds there, which is not a reference at all. brae scanned every patch, so on a case
                // with a non-uniform initial p the zeroGradient patches dragged the reference and the
                // limits came out different from OF's. Identical on a uniform initial p, which is why
                // every gate agreed.
                scalar pRefMax = -1e300;
                scalar pRefMin = 1e300;
                for (std::size_t pi = 0; pi < fvp.size(); ++pi)
                {
                    if (!p.boundary[pi]->fixesValue()) continue;
                    const std::vector<scalar>& bv = p.boundary[pi]->value();
                    for (scalar v : bv)
                    {
                        pRefMax = std::max(pRefMax, v);
                        pRefMin = std::min(pRefMin, v);
                    }
                }
                if (pRefMax <= -1e299)
                {
                    throw std::runtime_error(
                        "brae: SIMPLE/pMaxFactor or pMinFactor given, but NO pressure patch fixes a value, "
                        "so there is no reference pressure to scale. OpenFOAM refuses the same case "
                        "(pressureControl.C: \"pressure limits are not set\"). Specify absolute pMax/pMin "
                        "instead, or give the case a fixedValue pressure patch.");
                }
                if (rcm.pMaxFactor > 0.0) { rcm.pMaxLimit = pRefMax * rcm.pMaxFactor; rcm.limitMaxP = true; }
                if (rcm.pMinFactor > 0.0) { rcm.pMinLimit = pRefMin * rcm.pMinFactor; rcm.limitMinP = true; }
                std::printf("brae_rhoSimpleFoam: pressureControl limits p to [%g, %g] Pa\n",
                            rcm.limitMinP ? rcm.pMinLimit : -1e300,
                            rcm.limitMaxP ? rcm.pMaxLimit : 1e300);
            }
        }

        DeviceSimpleControls ctl;
        // The pressure needs a reference iff NO p patch fixes the value -- otherwise the all-Neumann
        // system is singular. gpuSimpleFoam has always done this scan; the compressible driver never did,
        // so needRef stayed at its default false and a closed compressible domain (all walls, or
        // fixedFluxPressure everywhere) would solve a singular pressure equation with no reference and
        // no adjustPhi. Same rule as the incompressible driver: a fixedValue p, or a freestreamPressure
        // (bcCategory 4, fixedValue on outflow), references the pressure.
        ctl.needRef = true;
        for (const auto& bf : p.boundary)
            if (bf->fixesValue() || bf->bcCategory() == 4)
            {
                ctl.needRef = false;
                break;
            }
        if (ctl.needRef)
        {
            // rc already carries these, read from the SIMPLE sub-dict by readRhoSimpleControls.
            ctl.pRefCell = rc.pRefCell;
            ctl.pRefValue = rc.pRefValue;
        }
        // ctl.nu is the KINEMATIC viscosity [m^2/s]; tc.mu0 is DYNAMIC [Pa s]. Assigning one to the other
        // was a units error off by rho -- ~1.2x for air at STP, but rho spans 0.87..1.16 even on the small
        // heated duct and far more on a real compressible case.
        //
        // The old comment said "only the seed matters", and that was true until E5: with `turbulence on`
        // the per-iteration solve overwrites nut from th_.mu, so the bad seed washes out in one iteration
        // and no gate could see it. With OF's `RAS { turbulence off; }` the model never runs, so the
        // startup correctNut IS the answer -- and its F2 (arg2 = max(2*sqrt(k)/(betaStar*omega*y),
        // 500*nu/(y^2*omega))) then uses mu as nu, activating the Bradshaw limiter where OF leaves it
        // inactive. Measured on rhoTI with the switch off: 40 inlet-column cells wrong by 75%.
        //
        // Seeded from the case's OWN initial state rather than a constant. Still a single scalar where OF
        // has a field -- validateTurbulence() runs before the thermo is seeded, so a per-cell nu is not
        // available there. Tracked as E7.
        {
            scalar rhoMean = 0.0;
            for (label i = 0; i < nC; ++i) rhoMean += p.internal[i] / (tc.R * T.internal[i]);
            rhoMean = (nC > 0) ? rhoMean / static_cast<scalar>(nC) : scalar(1);
            ctl.nu = tc.mu0 / rhoMean;
        }
        // B1: the transonic branch (phid, the phiHbyA subtraction, implicit fvm::div(phid,p) folded into
        // the pressure matrix, pEqn.relax(), and a BiCGStab solve because the resulting matrix is not
        // symmetric). It ran behind a refusal until it could be shown right on a case that discriminates.
        //
        // The refusal is now LIFTED, on this evidence:
        //   * squareBend -- the actual tutorial B1 exists for, Mach ~0.96, transonic AND consistent
        //     (SIMPLEC) -- converges in 160 iterations against OF's 156 and agrees to
        //     p 1.8e-03, U 1.4e-03, T 6.0e-04, rho 1.6e-03, k 7.4e-03, epsilon 8.3e-03, nut 4.9e-03.
        //     Gate `transonic_vs_openfoam`.
        //   * a low-Mach compressible duct converges in 105 against OF's 104 (p 7.0e-09, T 1.8e-07).
        //     That case does NOT discriminate -- the subsonic branch gives the same answer to 1e-11 --
        //     so it only shows the branch does not BREAK a subsonic case. It is not the evidence here.
        //
        // What actually made squareBend work was NOT a transonic fix. It diverged (contGlobal -2.83e+02
        // at iteration 1, NaN by 50) because of defects in the SHARED compressible path -- most recently
        // the thermo-type-dependent rho update below. The transonic assembly had been right for a while
        // behind an unrelated bug, which is the argument for fixing the chain before trusting a branch.
        const_cast<RhoSimpleControls&>(rc).rhoLagsPressure = tc.rhoThermoType;   // heRhoThermo rho timing
        ctl.transonic = rc.transonic;   // B1: OF pEqn.H transonic branch (guarded above)
        parseFvSchemesControls(caseDir, ctl);

        // Turbulence, through the SAME readers simpleFoam and pimpleFoam use, so a compressible case gets
        // exactly the model selection, coefficient set and wall-function guards an incompressible one does.
        // (turbProps itself is read above, next to the other dicts, so the audit guard can hold it.)
        const std::string simType = turbProps.wordOr("simulationType", "laminar");
        if (simType != "RAS" && simType != "laminar")
            throw std::runtime_error("brae: unsupported simulationType '" + simType + "' for rhoSimpleFoam (RAS or laminar)");
        ctl.turbulent = (simType == "RAS");
        readTurbulenceModel(turbProps, ctl);
        // kOmegaSST and kEpsilon are both rho-weighted (every RHS term, the diffusivity, the volumetric
        // divU and the per-face wall nu). SA and the kOmegaSST variants are not, so they stay refused:
        // running one down the incompressible path converges to a wrong answer rather than failing.
        // Standard kEpsilon = not SST, not SA, not realizableKE. realizableKE stays refused because its
        // epsilon reaction is a DIFFERENT expression (deviceEpsReactionRealizable, strain-based) that has
        // NOT been rho-weighted -- accepting it here would run an unweighted reaction and converge wrong.
        const bool keStandard = !ctl.sst && !ctl.sa && !ctl.keCoeffs.realizable;
        if (ctl.turbulent && !ctl.sst && !keStandard)
            throw std::runtime_error(
                "brae: rhoSimpleFoam supports kOmegaSST and kEpsilon so far. SpalartAllmaras and "
                "realizableKE are not rho-weighted, and running one down the incompressible path gives a "
                "converged wrong answer, so they are refused instead.");

        // div(phi,h|e) linearUpwind is HONOURED: measured against OF on the heated duct it agrees to
        // 8.2e-7, i.e. as well as upwind does. An earlier measurement said otherwise (7.1e-2) and was
        // wrong -- it used a one-line fvSchemes, which the old line-based parser let leak the energy
        // scheme onto div(phi,U). The parser now splits on statements, so layout cannot do that.
        //
        // Turbulence-scalar linearUpwind is HONOURED here. That opt-out was inherited from gpuSimpleFoam
        // on an unmeasured claim ("linearUpwind degrades k/omega vs OF"); measured on this path it is the
        // DOWNGRADE that is the error. validation/luturb_vs_openfoam.sh (the rhoSST duct with
        // div(phi,k|omega) = linearUpwind) against OF v2412, converged fields:
        //     honoured : k 2.0e-06   omega 3.2e-06   nut 8.1e-07
        //     upwind   : k 6.8e-03   omega 8.9e-03   nut 1.6e-02
        // so running upwind against a case that asked for linearUpwind cost 1.6% on nut.
        //
        // gpuSimpleFoam keeps its guard for a DIFFERENT reason -- pitzDaily SST diverges from a cold start
        // with linearUpwind on k -- and that is a convergence-path problem, not this discretisation: ONE
        // iteration from OF's own converged pitzDaily state agrees to 1.6e-06 on k. See the note there.
        // BRAE_SCALAR_LINEARUPWIND=0 forces upwind as an escape hatch.
        if (const char* luEnv = std::getenv("BRAE_SCALAR_LINEARUPWIND"))
        {
            if (std::atoi(luEnv) == 0) { ctl.luK = false; ctl.luEps = false; }
        }

        // OF looks these up by FIELD NAME: "omega" on kOmegaSST, "epsilon" on kEpsilon.
        // fvOptions. The compressible driver did not open this file AT ALL, where gpuSimpleFoam reads it
        // and refuses anything it cannot apply. angledDuctExplicitFixedCoeff carries an
        // explicitPorositySource that dominates its momentum balance, plus two constraints -- all three
        // vanished in silence and the case still converged, to a different problem. Refuse instead: brae
        // does not run a case with a dropped source term.
        {
            std::ifstream fa(caseDir + "/system/fvOptions"), fb(caseDir + "/constant/fvOptions");
            if (fa.good() || fb.good())
            {
                noticeIgnored("fvOptions", "found an fvOptions file; the compressible solver cannot apply "
                                           "any source or constraint yet");
                throw std::runtime_error(
                    "brae: this case has an fvOptions file, and brae_rhoSimpleFoam cannot apply fvOptions "
                    "sources/constraints. Running anyway would SILENTLY drop them (a porosity source, a "
                    "temperature constraint) and converge to a different problem. Remove or disable it, or "
                    "use the incompressible solver, which supports a subset.");
            }
        }

        const std::string second = ctl.sst ? "omega" : "epsilon";
        readRelaxationFactors(fvSolution, ctl);   // shared; adds the alpha<=0 guard this copy lacked

        // fvSolution -> ctl, through the SHARED reader. This driver previously read only tolP/tolU/tolKE
        // by hand and silently dropped relTol{P,U,KE}, consistent (SIMPLEC), nNonOrthogonalCorrectors,
        // the smoothSolver selection and the perf knobs -- see linear_solver_setup.cuh.
        readLinearSolverControls(fvSolution, second, ctl);
        const EnergySolverControls eSolve = readEnergySolverControls(fvSolution, tc.internalEnergy);

        TurbulenceFields tf;
        if (ctl.turbulent) tf = readTurbulenceFields(t0, fvp, nC, ctl, second, U);

        // Per-boundary-face Prt from 0/alphat. OF keeps two DIFFERENT turbulent Prandtl numbers in one
        // case: alphatWallFunction reads its own from the patch (default 0.85) while the turbulence model
        // uses the one from its coeffs dict (default 1.0). Using either one everywhere is wrong somewhere.
        std::vector<scalar> prtFace;
        if (ctl.turbulent)
        {
            FieldData<scalar> alphatFd;
            bool haveAlphat = true;
            try { alphatFd = readField<scalar>(t0 + "/alphat"); }
            catch (const std::exception&) { haveAlphat = false; }

            for (const FvPatch& q : fvp)
            {
                if (q.type == "cyclic" || q.type == "cyclicAMI") continue;   // DeviceBoundary skips these
                scalar prt = tc.Prt;   // the MODEL's Prt away from an alphat wall function
                // OF-style resolution (exact name, then group, then regex) -- NOT `pb.name == q.name`.
                // squareBend* key this entry as "(?i).*walls" against a patch literally called `walls`,
                // so exact matching missed it and Prt silently reverted to the model default 1.0 instead
                // of the wall function's 0.85: wall alphat, and the wall heat flux, ~15% low.
                if (haveAlphat)
                {
                    const PatchFieldData<scalar>* pb = findPatchEntry(alphatFd.boundary, q);
                    if (pb && (pb->type == "compressible::alphatWallFunction" || pb->type == "alphatWallFunction"))
                        prt = pb->Prt;
                }
                prtFace.insert(prtFace.end(), static_cast<std::size_t>(q.size), prt);
            }
        }

        const int endTime = static_cast<int>(controlDict.scalarOr("endTime", 1000));

        DeviceSimpleSolver solver(m, g, fvp, U, p, phi, ctl,
                                  ctl.turbulent ? &tf.k : nullptr,
                                  ctl.turbulent ? &tf.eps : nullptr,
                                  ctl.turbulent ? &tf.nut : nullptr);

        // he boundary: built from the case's 0/T, then converted. brae never reads a 0/he, exactly as OF
        // never asks a user to write one.
        DeviceBoundary dbT = buildDeviceBoundary(T, fvp, g);
        DeviceBoundary dbHe = buildDeviceBoundary(T, fvp, g);
        deviceEnergyBoundaryFromT(dbT, tc, dbHe);
        // OF re-evaluates the turbulent-inlet BCs every updateCoeffs; give the solver the per-face
        // masks so it refreshes them each iteration instead of freezing the set-up value.
        solver.setTurbulentInlets(tf.turbInletMasks.tiMask, tf.turbInletMasks.tiIntensity,
                                  tf.turbInletMasks.mlMask, tf.turbInletMasks.mlLength);
        solver.setCompressible(tc, rc, std::move(dbHe));
        solver.setAlphatPrt(prtFace);
        solver.setEnergySolver(eSolve.tol, eSolve.relTol, eSolve.useGS);

        // Seed the thermo: T from the case, he from T, then one thermo.correct() so rho/psi/mu/alpha are
        // consistent before the first momentum predictor, and rhoPrev so the first relax has a partner.
        DeviceThermo& th = solver.thermo();
        th.T.copyFrom(T.internal);
        deviceThermoHeFromT(th, tc);
        deviceThermoCorrect(th, solver.pDevice(), tc);
        // OF createFields.H: `volScalarField rho(IOobject("rho", ...), thermo.rho())`. rho is initialised
        // from thermo.rho() as its OWN statement, because hePsiThermo::calculate never writes a rho --
        // psiThermo has no rho_ field at all. Folding this into the correct() call is what the old
        // deviceThermoUpdate did, and it is exactly the conflation that made picking `updateRho` a guess.
        deviceThermoRho(th, solver.pDevice(), tc, th.rho);
        deviceRhoSeedPrev(th);
        // E7: OF validates the turbulence model AFTER the thermo is constructed; brae's solver ctor did it
        // before, so correctNut ran against a placeholder inlet density. Redo it now that rho is real.
        solver.revalidateAfterThermo();

        // What brae RESOLVED from the case, not what the dict says. A relaxation factor silently left at
        // 1.0 is invisible in the fields and fatal on a stiff case.
        std::printf("brae_rhoSimpleFoam: solve  relTol p=%g U=%g k/%s=%g e|h=%g   tol p=%g U=%g e|h=%g   "
                    "GS U=%d k=%d %s=%d e|h=%d   SIMPLEC=%d nNonOrth=%d\n",
                    ctl.relTolP, ctl.relTolU, second.c_str(), ctl.relTolKE, eSolve.relTol,
                    ctl.tolP, ctl.tolU, eSolve.tol,
                    (int)ctl.gsU, (int)ctl.gsK, second.c_str(), (int)ctl.gsEps, (int)eSolve.useGS,
                    (int)ctl.consistent, ctl.nNonOrth);
        std::printf("brae_rhoSimpleFoam: relax  p=%g U=%g e|h=%g k=%g %s=%g rho=%g   pLimit=[%g, %g]\n",
                    ctl.relaxP, ctl.relaxU, rc.relaxHe, ctl.relaxK, second.c_str(), ctl.relaxEps, tc.relaxRho,
                    rc.limitMinP ? rc.pMinLimit : -1.0, rc.limitMaxP ? rc.pMaxLimit : -1.0);
        std::printf("brae_rhoSimpleFoam: %ld cells, subsonic %s, R=%.3f Cp=%.1f\n",
                    (long)nC, ctl.turbulent ? (ctl.sst ? "kOmegaSST" : "kEpsilon") : "laminar", tc.R, tc.Cp);

        // C5: SIMPLE residualControl. The compressible driver had none at all -- it always ran to endTime,
        // so a case asking to stop at p 1e-4 burned every remaining iteration and, more importantly, brae
        // reported a different iteration count than OF for the same input while claiming to run the same
        // case. OF's rule (empty dict never converges; `achieved && checked`) lives in the shared
        // ResidualControl, and its dict lookup is regex-aware, which matters here: every stock tutorial
        // writes its turbulence criteria as a pattern, e.g. `"(k|omega|e)" 1e-4`.
        const FoamDict* simpleDict = fvSolution.subDict("SIMPLE");
        ResidualControl resControl(simpleDict ? simpleDict->subDict("residualControl") : nullptr);
        // OF names the energy field by the case's own energy variable; the solve reports it as "he".
        const std::string heName = tc.internalEnergy ? "e" : "h";
        std::printf("  residualControl=%s\n", resControl.active() ? "on" : "off");

        // OF controlDict write cadence. dict_audit found that this driver read NONE of writeControl /
        // writeInterval / purgeWrite / deltaT / stopAt: it wrote exactly one time directory, at the end,
        // so a case asking to write every N iterations silently got nothing until convergence. The policy
        // is shared with gpuSimpleFoam (write_control.cuh); only the payload below is solver-specific.
        WriteControl wc(controlDict);
        const std::string wsrc = t0 + "/";
        auto writeTimeDir = [&](const std::string& tname)
        {
            const std::string outDir = caseDir + "/" + tname;
            std::filesystem::create_directories(outDir);
            writeVolField(wsrc + "U", outDir + "/U", solver.U(), fvp, 12, solver.UBoundary());
            writeVolField(wsrc + "p", outDir + "/p", solver.p(), fvp, 12, solver.pBoundary());
            {
                std::vector<scalar> Tout(nC);
                th.T.copyTo(Tout);
                writeVolField(wsrc + "T", outDir + "/T", Tout, fvp, 12, solver.TBoundary());
            }
            {
                // rho, so the gate can compare the EOS result directly rather than inferring it from p and T.
                // 0/T is only a TEMPLATE for the FoamFile header here -- the identity, the dimensions and every
                // boundary entry are declared, not inherited. Written from T alone, rho came out as `object T`,
                // dimensions of temperature, an inlet density of 300 and (once B5 landed) a fixedGradient
                // density of 20000 kg/m^4, all of which OF reads back without complaint.
                static const DerivedFieldSpec rhoSpec{"rho", "dimensions      [1 -3 0 0 0 0 0];"};
                std::vector<scalar> rhoOut(nC);
                th.rho.copyTo(rhoOut);
                writeVolField(wsrc + "T", outDir + "/rho", rhoOut, fvp, 12, solver.rhoBoundary(), &rhoSpec);
            }
            if (ctl.turbulent)
            {
                writeVolField(wsrc + "k", outDir + "/k", solver.k(), fvp, 12, solver.kBoundary());
                // the 2nd turbulence scalar shares one slot: omega on kOmegaSST, epsilon on kEpsilon
                writeVolField(wsrc + second, outDir + "/" + second, solver.eps(), fvp, 12, solver.epsBoundary());
                writeVolField(wsrc + "nut", outDir + "/nut", solver.nut(), fvp, 12, solver.nutBoundary());
            }
            std::printf("written %s/{U,p,T,rho%s}\n", outDir.c_str(),
                        ctl.turbulent ? (ctl.sst ? ",k,omega,nut" : ",k,epsilon,nut") : "");
            wc.recordWritten(caseDir, tname);
        };

        int nIter = endTime;
        bool converged = false;
        for (int iter = 1; iter <= endTime; ++iter)
        {
            clearTurbulenceReport();
            const DeviceSimpleResidual r = solver.rhoSimpleStep();
            if (iter % 50 == 0 || iter == 1)
            {
                std::printf("Time = %d   Ux %.4e  p %.4e  contGlobal %.4e\n",
                            iter, r.Ux, r.p, r.contGlobal);
            }
            resControl.beginIteration();
            // U is gated on Ux alone, matching gpuSimpleFoam: brae tracks no solved-directions mask, so the
            // out-of-plane component of a 2D/empty or wedge case carries a degenerate residual that would
            // wrongly block convergence on every 2D case.
            bool achieved = resControl.ok(r.p, "p") && resControl.ok(r.Ux, "U");
            if (achieved)
                for (const auto& e : turbulenceReport())
                    if (!resControl.ok(e.perf.initialResidual, e.field == "he" ? heName : e.field)) { achieved = false; break; }
            if (resControl.converged(achieved)) { converged = true; nIter = iter; break; }
            // Intermediate write. Skipped on the last iteration, which the final write below covers.
            const scalar tval = wc.timeValue(iter);
            if (iter != endTime && wc.isWriteTime(iter, tval)) writeTimeDir(WriteControl::timeName(tval));
        }
        std::printf(converged ? "SIMPLE solution converged in %d iterations\n"
                              : "SIMPLE reached endTime (%d iterations)\n", nIter);

        // Always write the final (converged / endTime) state, as OF's writeAndEnd does. Named from the
        // TIME VALUE, not the iteration count, so a case with deltaT != 1 gets OF's directory names.
        const std::string finalName = WriteControl::timeName(wc.timeValue(nIter));
        writeTimeDir(finalName);
        std::printf("brae_rhoSimpleFoam: wrote %s\n", (caseDir + "/" + finalName).c_str());
        return 0;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
