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
#include <cstdio>
#include <string>
#include <vector>
#include <stdexcept>
#include <filesystem>

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
        const RhoSimpleControls rc = readRhoSimpleControls(fvSolution);
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

        DeviceSimpleControls ctl;
        ctl.nu = tc.mu0;                       // replaced every iteration by th_.mu; only the seed matters
        ctl.turbulent = false;                 // laminar until phase 4
        const int endTime = static_cast<int>(controlDict.scalarOr("endTime", 1000));

        DeviceSimpleSolver solver(m, g, fvp, U, p, phi, ctl);

        // he boundary: built from the case's 0/T, then converted. brae never reads a 0/he, exactly as OF
        // never asks a user to write one.
        DeviceBoundary dbT = buildDeviceBoundary(T, fvp, g);
        DeviceBoundary dbHe = buildDeviceBoundary(T, fvp, g);
        deviceEnergyBoundaryFromT(dbT, tc, dbHe);
        solver.setCompressible(tc, rc, std::move(dbHe));

        // Seed the thermo: T from the case, he from T, then one thermo.correct() so rho/psi/mu/alpha are
        // consistent before the first momentum predictor, and rhoPrev so the first relax has a partner.
        DeviceThermo& th = solver.thermo();
        th.T.copyFrom(T.internal);
        deviceThermoHeFromT(th, tc);
        deviceThermoUpdate(th, solver.pDevice(), tc);
        deviceRhoSeedPrev(th);

        std::printf("brae_rhoSimpleFoam: %ld cells, subsonic laminar, R=%.3f Cp=%.1f\n",
                    (long)nC, tc.R, tc.Cp);

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
        std::printf("brae_rhoSimpleFoam: wrote %s\n", outDir.c_str());
        return 0;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
