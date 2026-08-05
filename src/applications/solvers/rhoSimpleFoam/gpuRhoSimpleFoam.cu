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
#include "scheme_parse.cuh"        // parseFvSchemesControls
#include <cstdio>
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
                scalar pRefMax = -1e300;
                scalar pRefMin = 1e300;
                for (std::size_t pi = 0; pi < fvp.size(); ++pi)
                {
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
                        "brae: SIMPLE/pMaxFactor or pMinFactor given, but no boundary pressure to scale "
                        "them against. Specify absolute pMax/pMin instead (OF refuses the same case).");
                }
                if (rcm.pMaxFactor > 0.0) { rcm.pMaxLimit = pRefMax * rcm.pMaxFactor; rcm.limitMaxP = true; }
                if (rcm.pMinFactor > 0.0) { rcm.pMinLimit = pRefMin * rcm.pMinFactor; rcm.limitMinP = true; }
                std::printf("brae_rhoSimpleFoam: pressureControl limits p to [%g, %g] Pa\n",
                            rcm.limitMinP ? rcm.pMinLimit : -1e300,
                            rcm.limitMaxP ? rcm.pMaxLimit : 1e300);
            }
        }

        DeviceSimpleControls ctl;
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
        if (ctl.turbulent && !ctl.sst)
            throw std::runtime_error(
                "brae: rhoSimpleFoam supports kOmegaSST only so far. The other RAS models are not yet "
                "rho-weighted, and running one down the incompressible path gives a converged wrong answer.");

        // div(phi,h|e) linearUpwind is HONOURED: measured against OF on the heated duct it agrees to
        // 8.2e-7, i.e. as well as upwind does. An earlier measurement said otherwise (7.1e-2) and was
        // wrong -- it used a one-line fvSchemes, which the old line-based parser let leak the energy
        // scheme onto div(phi,U). The parser now splits on statements, so layout cannot do that.
        //
        // The turbulence scalars keep gpuSimpleFoam's opt-out: that gate predates this work and its
        // claim (linearUpwind degrades k/omega vs OF) has not been re-measured here, so it is left as
        // found rather than flipped on an untested assumption.
        if (!std::getenv("BRAE_SCALAR_LINEARUPWIND"))
        {
            if (ctl.luK || ctl.luEps)
            {
                std::fprintf(stderr,
                    "brae WARNING: fvSchemes requests 'linearUpwind' on div(phi,k|omega) but brae is "
                    "running UPWIND there. Set BRAE_SCALAR_LINEARUPWIND=1 to honour it. The energy "
                    "equation honours linearUpwind unconditionally (validated against OpenFOAM).\n");
            }
            ctl.luK = false;
            ctl.luEps = false;
        }

        const FoamDict* rf = fvSolution.subDict("relaxationFactors");
        const FoamDict* eqs = rf ? rf->subDict("equations") : nullptr;
        const FoamDict* fld = rf ? rf->subDict("fields") : nullptr;
        const FoamDict* eqSrc = eqs ? eqs : rf;
        const FoamDict* fldSrc = fld ? fld : rf;
        ctl.relaxU = eqSrc ? eqSrc->scalarOr("U", 1.0) : 1.0;
        ctl.relaxK = eqSrc ? eqSrc->scalarOr("k", 1.0) : 1.0;
        ctl.relaxEps = eqSrc ? eqSrc->scalarOr("omega", 1.0) : 1.0;
        ctl.relaxP = fldSrc ? fldSrc->scalarOr("p", 1.0) : 1.0;

        const FoamDict* solvers = fvSolution.subDict("solvers");
        auto solverTol = [&](const std::string& f, scalar def)
        {
            const FoamDict* sd = solvers ? solvers->subDict(f) : nullptr;
            return sd ? sd->scalarOr("tolerance", def) : def;
        };
        ctl.tolP = solverTol("p", 1e-6);
        ctl.tolU = solverTol("U", 1e-8);
        ctl.tolKE = std::fmin(solverTol("k", 1e-8), solverTol("omega", 1e-8));

        TurbulenceFields tf;
        if (ctl.turbulent) tf = readTurbulenceFields(t0, fvp, nC, ctl, "omega", U);

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
                if (haveAlphat)
                    for (const PatchFieldData<scalar>& pb : alphatFd.boundary)
                        if (pb.name == q.name
                         && (pb.type == "compressible::alphatWallFunction" || pb.type == "alphatWallFunction"))
                            prt = pb.Prt;
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
        solver.setCompressible(tc, rc, std::move(dbHe));
        solver.setAlphatPrt(prtFace);

        // Seed the thermo: T from the case, he from T, then one thermo.correct() so rho/psi/mu/alpha are
        // consistent before the first momentum predictor, and rhoPrev so the first relax has a partner.
        DeviceThermo& th = solver.thermo();
        th.T.copyFrom(T.internal);
        deviceThermoHeFromT(th, tc);
        deviceThermoUpdate(th, solver.pDevice(), tc);
        deviceRhoSeedPrev(th);

        std::printf("brae_rhoSimpleFoam: %ld cells, subsonic %s, R=%.3f Cp=%.1f\n",
                    (long)nC, ctl.turbulent ? "kOmegaSST" : "laminar", tc.R, tc.Cp);

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
        writeVolField(wsrc + "p", outDir + "/p", solver.p(), fvp, 12);
        {
            std::vector<scalar> Tout(nC);
            th.T.copyTo(Tout);
            writeVolField(wsrc + "T", outDir + "/T", Tout, fvp, 12);
        }
        {
            // rho, so the gate can compare the EOS result directly rather than inferring it from p and T.
            std::vector<scalar> rhoOut(nC);
            th.rho.copyTo(rhoOut);
            writeVolField(wsrc + "T", outDir + "/rho", rhoOut, fvp, 12);
        }
        if (ctl.turbulent)
        {
            writeVolField(t0 + "/k",     outDir + "/k",     solver.k(),   fvp, 12);
            writeVolField(t0 + "/omega", outDir + "/omega", solver.eps(), fvp, 12);   // de_ slot holds omega on SST
            writeVolField(t0 + "/nut",   outDir + "/nut",   solver.nut(), fvp, 12);
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
