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

        const ThermoCoeffs tc = readThermoCoeffs(caseDir);
        const RhoSimpleControls rc = readRhoSimpleControls(fvSolution, tc.internalEnergy);
        if (rc.transonic)
        {
            throw std::runtime_error(
                "brae: SIMPLE/transonic yes is not implemented. brae's rhoSimpleFoam is subsonic only "
                "(the transonic branch adds fvm::div(phid,p), which is not in this build). Running a "
                "transonic case down the subsonic branch converges to a wrong answer silently, so it is "
                "refused instead.");
        }

        PrimitiveMesh m;
        m.read(caseDir + "/constant/polyMesh");
        FvGeometry g;
        g.build(m);
        const std::vector<FvPatch> fvp = buildPatches(m, g);
        const label nC = m.nCells();

        const std::string t0 = caseDir + "/0";
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
        ctl.nu = tc.mu0;                       // replaced every iteration by th_.mu; only the seed matters
        parseFvSchemesControls(caseDir, ctl);

        // Turbulence, through the SAME readers simpleFoam and pimpleFoam use, so a compressible case gets
        // exactly the model selection, coefficient set and wall-function guards an incompressible one does.
        const FoamDict turbProps = readDict(caseDir + "/constant/turbulenceProperties");
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
        deviceThermoUpdate(th, solver.pDevice(), tc);
        deviceRhoSeedPrev(th);

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

        for (int iter = 1; iter <= endTime; ++iter)
        {
            const DeviceSimpleResidual r = solver.rhoSimpleStep();
            if (iter % 50 == 0 || iter == 1)
            {
                std::printf("Time = %d   Ux %.4e  p %.4e  contGlobal %.4e\n",
                            iter, r.Ux, r.p, r.contGlobal);
            }
        }

        const std::string outDir = caseDir + "/" + std::to_string(endTime);
        std::filesystem::create_directories(outDir);
        const std::string wsrc = t0 + "/";
        writeVolField(wsrc + "U", outDir + "/U", solver.U(), fvp, 12);
        writeVolField(wsrc + "p", outDir + "/p", solver.p(), fvp, 12, solver.pBoundary());
        {
            std::vector<scalar> Tout(nC);
            th.T.copyTo(Tout);
            writeVolField(wsrc + "T", outDir + "/T", Tout, fvp, 12, solver.TBoundary());
        }
        {
            // rho, so the gate can compare the EOS result directly rather than inferring it from p and T.
            std::vector<scalar> rhoOut(nC);
            th.rho.copyTo(rhoOut);
            writeVolField(wsrc + "T", outDir + "/rho", rhoOut, fvp, 12);
        }
        if (ctl.turbulent)
        {
            writeVolField(t0 + "/k",     outDir + "/k",     solver.k(),   fvp, 12, solver.kBoundary());
            // the 2nd turbulence scalar shares one slot: omega on kOmegaSST, epsilon on kEpsilon
            writeVolField(t0 + "/" + second, outDir + "/" + second, solver.eps(), fvp, 12, solver.epsBoundary());
            writeVolField(t0 + "/nut",   outDir + "/nut",   solver.nut(), fvp, 12, solver.nutBoundary());
        }
        std::printf("brae_rhoSimpleFoam: wrote %s\n", outDir.c_str());
        return 0;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
