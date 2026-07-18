#pragma once
// brae distributed DEVICE simpleFoam: the multi-GPU app path.
//
//   mpirun -np N brae -case <dir> -parallel
//
// One rank == one partition == one GPU. Mirrors the control flow of the host brae_simpleFoam distributed path
// (simpleFoam.cu) -- read the global case on every rank, decompose in core with SCOTCH (no separate
// decomposePar step), iterate, gather on write -- but the SIMPLE loop itself runs entirely on the GPU via
// ParallelDeviceSimple: U/p/phi stay device-resident across iterations and only cross PCIe at write time.
//
// Laminar OR RAS (kEpsilon, kOmegaSST): the SIMPLE loop runs on the GPU via ParallelDeviceSimple, with
// turbulence enabled per the case's turbulenceProperties. Unsupported models/schemes are refused, not approximated.
//
// The single-GPU `brae` path is untouched: this is reached only when -parallel is passed, so a plain
// `brae -case <dir>` never initialises MPI (which also keeps the -cases fork orchestrator clear of MPI).
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "foam_field_writer.cuh"
#include "foam_dict.cuh"
#include "fvc.cuh"
#include "scotch_decomposition.cuh"
#include "parallel_simple.cuh"
#include "field_distribute.cuh"
#include "parallel_device_simple.cuh"
#include "cf_pstream.cuh"

#include <cmath>
#include <cstdio>
#include <sstream>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>
#include <chrono>
#include <cstdlib>

namespace brae {

namespace detail {

// Gather a partitioned internal field to the global cell ordering. Partitions own disjoint cells, so a
// scatter-into-zeros + sum reduce reconstructs the global field on every rank (decomposePar's inverse).
inline std::vector<scalar> gatherToGlobal(const Partition& P, const std::vector<scalar>& loc)
{
    std::vector<scalar> g(P.globalNCells, 0.0);
    for (label c = 0; c < P.nCells(); ++c)
        g[P.Lm.cellProcAddr[c]] = loc[c];
    Pstream::allReduce(g.data(), static_cast<int>(g.size()), ReduceOp::Sum);
    return g;
}

inline std::vector<vector> gatherToGlobal(const Partition& P, const std::vector<vector>& loc)
{
    std::vector<scalar> g(3 * static_cast<std::size_t>(P.globalNCells), 0.0);
    for (label c = 0; c < P.nCells(); ++c)
    {
        const label gc = P.Lm.cellProcAddr[c];
        g[3 * gc]     = loc[c].x;
        g[3 * gc + 1] = loc[c].y;
        g[3 * gc + 2] = loc[c].z;
    }
    Pstream::allReduce(g.data(), static_cast<int>(g.size()), ReduceOp::Sum);
    std::vector<vector> out(P.globalNCells);
    for (label c = 0; c < P.globalNCells; ++c)
        out[c] = vector{g[3 * c], g[3 * c + 1], g[3 * c + 2]};
    return out;
}


} // namespace detail

// The `brae ... -parallel` entry point. Returns a process exit code.
inline int runParallelDeviceFoam(int argc, char** argv)
{
    Pstream::init(argc, argv);
    const bool master = Pstream::master();
    const int  nproc  = Pstream::nProcs();
    const int  rank   = Pstream::myProcNo();
    int rc = 0;
    try
    {
        std::string caseDir = ".";
        for (int i = 1; i < argc; ++i)
        {
            const std::string a = argv[i];
            if (a == "-case" && i + 1 < argc) caseDir = argv[++i];
            else if (a[0] != '-')             caseDir = a;
        }

        if (!Pstream::nvshmemActive() && nproc > 1)
            throw std::runtime_error(
                "brae -parallel: NVSHMEM is not active, so there is no GPU-to-GPU transport. Rebuild with "
                "-DBRAE_WITH_NVSHMEM=ON and check `nvidia-smi topo -p2p r` reports OK (P2P is required).");

        const FoamDict controlDict = readDict(caseDir + "/system/controlDict");
        const FoamDict fvSolution  = readDict(caseDir + "/system/fvSolution");
        const FoamDict transport   = readDict(caseDir + "/constant/transportProperties");
        const FoamDict turbProps   = readDict(caseDir + "/constant/turbulenceProperties");

        const std::string startStr = controlDict.wordOr("startTime", "0");
        const std::string fieldDir = caseDir + "/" + startStr;
        const int   endTime   = controlDict.intOr("endTime", 1000);
        const int   precision = controlDict.intOr("writePrecision", 16);
        const scalar nu       = transport.scalarOr("nu", 1e-5);

        // Turbulence: laminar, or RAS kEpsilon (the loop-integrated + validated model). kOmegaSST correct() is
        // validated standalone (test_gpu_parallel_komegasst) but not yet wired into the distributed loop, so it
        // is refused here -- honestly, not silently solved as something else.
        const std::string simType = turbProps.wordOr("simulationType", "laminar");
        const bool turbulent = (simType == "RAS");
        KEpsilonCoeffs keCoeffs;
        KOmegaSSTCoeffs sstCoeffs;
        bool sst = false;
        if (turbulent)
        {
            const FoamDict* ras = turbProps.subDict("RAS");
            const std::string model = ras ? ras->wordOr("RASModel", "") : "";
            if (model != "kEpsilon" && model != "kOmegaSST")
                throw std::runtime_error(
                    "brae -parallel: RASModel '" + model + "' is not implemented on the multi-GPU device path "
                    "(kEpsilon and kOmegaSST are). Use `mpirun -np N brae_simpleFoam -case <dir>` (host) or "
                    "`brae -case <dir>` (single GPU) for other models.");
            sst = (model == "kOmegaSST");
            const FoamDict* kec = ras ? ras->subDict("kEpsilonCoeffs") : nullptr;
            if (kec)
            {
                keCoeffs.Cmu = kec->scalarOr("Cmu", keCoeffs.Cmu);
                keCoeffs.C1 = kec->scalarOr("C1", keCoeffs.C1);
                keCoeffs.C2 = kec->scalarOr("C2", keCoeffs.C2);
                keCoeffs.C3 = kec->scalarOr("C3", keCoeffs.C3);
                keCoeffs.sigmaK = kec->scalarOr("sigmak", keCoeffs.sigmaK);
                keCoeffs.sigmaEps = kec->scalarOr("sigmaEps", keCoeffs.sigmaEps);
                keCoeffs.kappa = kec->scalarOr("kappa", keCoeffs.kappa);
                keCoeffs.E = kec->scalarOr("E", keCoeffs.E);
            }
            const FoamDict* ssc = ras ? ras->subDict("kOmegaSSTCoeffs") : nullptr;
            if (ssc)
            {
                sstCoeffs.alphaK1 = ssc->scalarOr("alphaK1", sstCoeffs.alphaK1);
                sstCoeffs.alphaK2 = ssc->scalarOr("alphaK2", sstCoeffs.alphaK2);
                sstCoeffs.alphaOmega1 = ssc->scalarOr("alphaOmega1", sstCoeffs.alphaOmega1);
                sstCoeffs.alphaOmega2 = ssc->scalarOr("alphaOmega2", sstCoeffs.alphaOmega2);
                sstCoeffs.gamma1 = ssc->scalarOr("gamma1", sstCoeffs.gamma1);
                sstCoeffs.gamma2 = ssc->scalarOr("gamma2", sstCoeffs.gamma2);
                sstCoeffs.beta1 = ssc->scalarOr("beta1", sstCoeffs.beta1);
                sstCoeffs.beta2 = ssc->scalarOr("beta2", sstCoeffs.beta2);
                sstCoeffs.betaStar = ssc->scalarOr("betaStar", sstCoeffs.betaStar);
                sstCoeffs.a1 = ssc->scalarOr("a1", sstCoeffs.a1);
                sstCoeffs.b1 = ssc->scalarOr("b1", sstCoeffs.b1);
                sstCoeffs.c1 = ssc->scalarOr("c1", sstCoeffs.c1);
            }
        }
        else if (simType != "laminar")
            throw std::runtime_error(
                "brae -parallel: simulationType '" + simType + "' is not supported (laminar or RAS).");

        // fvSchemes div(phi,U). The distributed path implements Gauss upwind, bounded, and linearUpwind (the
        // matrix is upwind either way; linearUpwind adds the deferred gradient correction, cut faces included).
        // Anything else -- limitedLinear, LUST, linearUpwindV, plain Gauss linear -- is a DIFFERENT
        // discretisation, and silently substituting upwind produces a converged, plausible, WRONG answer (on
        // the cavity that was a ~6% field difference). Refuse those, exactly as RAS is refused above.
        bool boundedDiv = false, linUpwind = false, lust = false, lapCorrected = false;
        {
            std::string divLine, lapLine;
            std::istringstream fsch(readFileExpanded(caseDir + "/system/fvSchemes"));
            std::string ln;
            while (std::getline(fsch, ln))
            {
                if (ln.find("div(phi,U)") != std::string::npos)      divLine = ln;
                if (ln.find("laplacianSchemes") != std::string::npos) lapLine = ln;
            }
            if (!divLine.empty())
            {
                boundedDiv = divLine.find("bounded") != std::string::npos;
                linUpwind  = divLine.find("linearUpwind") != std::string::npos;
                lust       = divLine.find("LUST") != std::string::npos;   // 0.75*linear + 0.25*linearUpwind (OF LUST.H)
                // NB "linearUpwind" has a capital U, so it does NOT contain the substring "upwind" -- the two
                // must be tested separately or a linearUpwind case reads as "no upwind scheme at all".
                const bool upwindFamily = (divLine.find("upwind") != std::string::npos) || linUpwind || lust;
                // linearUpwindV adds OF's vector direction limiter on top of linearUpwind -- NOT implemented,
                // and it contains the substring "linearUpwind", so it must be excluded explicitly.
                const bool unsupported = divLine.find("linearUpwindV") != std::string::npos
                                      || divLine.find("limitedLinear") != std::string::npos;
                if (!upwindFamily || unsupported)
                    throw std::runtime_error(
                        "brae -parallel: div(phi,U) scheme '" + divLine + "' is not implemented on the "
                        "multi-GPU device path (it supports 'Gauss upwind', 'bounded Gauss upwind', "
                        "'[bounded] Gauss linearUpwind grad(U)' and '[bounded] Gauss LUST grad(U)'). "
                        "Substituting a scheme would converge to a "
                        "DIFFERENT answer than fvSchemes asks for, so this is refused rather than solved "
                        "wrongly. Use `brae -case <dir>` (single GPU) for the full scheme set.");
            }
            // OF laplacian "corrected"/"limited" = nonOrthDeltaCoeffs implicit + an explicit corrVec.grad
            // correction. Checked against the mesh below, once the Partition exists.
            lapCorrected = (lapLine.find("corrected") != std::string::npos)
                        || (lapLine.find("limited")   != std::string::npos);
        }

        const FoamDict* rf  = fvSolution.subDict("relaxationFactors");
        const FoamDict* eqs = rf ? rf->subDict("equations") : nullptr;
        const FoamDict* fld = rf ? rf->subDict("fields") : nullptr;
        const scalar relaxU = eqs ? eqs->scalarOr("U", 1.0) : 1.0;
        const scalar relaxP = fld ? fld->scalarOr("p", 1.0) : 1.0;
        const scalar relaxK   = eqs ? eqs->scalarOr("k", 1.0) : 1.0;
        const scalar relaxEps = eqs ? eqs->scalarOr("epsilon", 1.0) : 1.0;

        const FoamDict* solvers = fvSolution.subDict("solvers");
        auto solverTol = [&](const std::string& f, scalar def)
        {
            const FoamDict* s = solvers ? solvers->subDict(f) : nullptr;
            return s ? s->scalarOr("tolerance", def) : def;
        };
        const scalar tolP = solverTol("p", 1e-6), tolU = solverTol("U", 1e-8);

        const FoamDict* simple = fvSolution.subDict("SIMPLE");
        // nNonOrthogonalCorrectors: OF re-solves the pEqn (n+1) times, recomputing the explicit non-orth
        // correction from the UPDATED p each pass. The distributed path does ONE pass (pass 0, the entry
        // grad(p)), so a case asking for more would silently get a different pressure. Refuse.
        if (simple && simple->intOr("nNonOrthogonalCorrectors", 0) > 0)
            throw std::runtime_error(
                "brae -parallel: nNonOrthogonalCorrectors > 0 is not implemented on the multi-GPU device path "
                "(it does a single pEqn pass). Set it to 0, or use `brae -case <dir>` (single GPU).");
        const FoamDict* resCtl = simple ? simple->subDict("residualControl") : nullptr;
        const bool   hasRC = (resCtl != nullptr);
        const scalar rcP = resCtl ? resCtl->scalarOr("p", -1) : -1;
        const scalar rcU = resCtl ? resCtl->scalarOr("U", -1) : -1;

        // Startup phase timing (BRAE_TIME_STARTUP=1): the ~17-20s fixed cost dominates the 2-GPU wall on a short
        // run, so break it down before optimising. cudaDeviceSynchronize before each lap captures GPU work; timed
        // on master (a rough per-phase breakdown, enough to find the dominant contributor).
        const bool timeStartup = std::getenv("BRAE_TIME_STARTUP") != nullptr;
        auto tclk = std::chrono::high_resolution_clock::now();
        auto lap = [&](const char* name)
        {
            if (!timeStartup) return;
            cudaDeviceSynchronize();
            auto now = std::chrono::high_resolution_clock::now();
            if (master) std::printf("[startup] %-14s %8.1f ms\n", name,
                                    std::chrono::duration<double, std::milli>(now - tclk).count());
            tclk = now;
        };

        PrimitiveMesh gm;
        gm.read(caseDir + "/constant/polyMesh");
        const label nC = gm.nCells();
        lap("meshRead");

        // decompose in core: SCOTCH on master, broadcast so every rank agrees on the same partitioning
        std::vector<label> cellToPart(nC, 0);
        if (master) cellToPart = scotchDecomposeCached(gm, nproc, caseDir + "/constant/polyMesh");   // ~1s -> cached
        Pstream::broadcast(cellToPart.data(), nC, 0);
        const Partition P(gm, cellToPart, rank);
        lap("decompose");

        // The non-orthogonal correction is NOT implemented at processor faces. On an ORTHOGONAL mesh it is
        // identically zero, so "corrected" is safe there and 15/113 cases say "orthogonal" outright. On a
        // non-orthogonal mesh, silently dropping it solves a DIFFERENT equation than fvSchemes asks for --
        // the same failure mode as substituting a div scheme, and 96/113 cases ask for "corrected". Refuse.
        // 0.1 deg: below that the correction is numerical noise; above it, it is physics we are not doing.
        if (lapCorrected)
        {
            const scalar nonOrtho = maxNonOrthogonality(P);
            // The non-orthogonal correction IS implemented on the distributed path: nonOrthDeltaCoeffs
            // implicit at cut faces (computeProcNonOrth), the explicit corrVec.grad correction for the
            // momentum, and for the pEqn BOTH the source and the face-flux correction (or continuity breaks).
            // Gated by test_gpu_parallel_duct on a sheared 26.57 deg mesh; every piece teeth-proven.
            if (master)
                std::printf("  mesh max non-orthogonality %.4g deg -> 'corrected' term negligible (< 0.1 deg)\n",
                            nonOrtho);
        }

        const FieldData<vector> Ufd = readField<vector>(fieldDir + "/U");
        const FieldData<scalar> pfd = readField<scalar>(fieldDir + "/p");
        GeometricField<vector> U0 = distributeField<vector>(Ufd, gm.patches(), P.Lm, P.lp, P.procW, rank);
        U0.evaluateBoundary();
        GeometricField<scalar> p0 = distributeField<scalar>(pfd, gm.patches(), P.Lm, P.lp, P.procW, rank);
        p0.evaluateBoundary();
        lap("distribute");

        // turbulence start fields (RAS kEpsilon)
        FieldData<scalar> kfd, efd, nfd;
        GeometricField<scalar> k0, eps0, nut0;
        if (turbulent)
        {
            kfd = readField<scalar>(fieldDir + "/k");
            efd = readField<scalar>(fieldDir + "/" + (sst ? "omega" : "epsilon"));
            nfd = readField<scalar>(fieldDir + "/nut");
            k0   = distributeField<scalar>(kfd, gm.patches(), P.Lm, P.lp, P.procW, rank); k0.evaluateBoundary();
            eps0 = distributeField<scalar>(efd, gm.patches(), P.Lm, P.lp, P.procW, rank); eps0.evaluateBoundary();
            nut0 = distributeField<scalar>(nfd, gm.patches(), P.Lm, P.lp, P.procW, rank); nut0.evaluateBoundary();
        }

        if (master)
        {
            std::printf("brae (device, distributed) | case=%s np=%d | %s%s | nu=%.3g | %ld cells\n",
                        caseDir.c_str(), nproc, simType.c_str(), turbulent ? (sst ? " (kOmegaSST)" : " (kEpsilon)") : "", nu, (long)nC);
            std::printf("  relax U=%.2g p=%.2g%s | tol p=%.1g U=%.1g | endTime=%d | residualControl=%s | div(phi,U)=%s\n",
                        relaxU, relaxP, turbulent ? (" k=" + std::to_string(relaxK)).c_str() : "",
                        tolP, tolU, endTime, hasRC ? "on" : "off",
                        (std::string(boundedDiv ? "bounded " : "") + (lust ? "Gauss LUST" : linUpwind ? "Gauss linearUpwind" : "Gauss upwind")).c_str());
        }

        std::vector<vector> Ug;
        std::vector<scalar> pg, kg, eg, ng;
        int nIter = 0;
        {   // scope: the solver's symmetric-heap buffers must be freed BEFORE Pstream::finalize
            ParallelDeviceSimple solver(P, U0, p0, nu, relaxU, relaxP, tolU, tolP, 2000, boundedDiv, linUpwind, lust);
            if (turbulent && sst) solver.enableTurbulenceSST(U0, k0, eps0, nut0, sstCoeffs, relaxK, relaxEps);
            else if (turbulent)   solver.enableTurbulence(U0, k0, eps0, nut0, keCoeffs, relaxK, relaxEps);
            lap("buildSolver");   // buildDeviceMesh + buildAMG + NVSHMEM halo/symmetric-heap alloc

            auto ok = [](scalar res, scalar ctl) { return ctl < 0 || res < ctl; };
            bool converged = false;
            int iter = 0;
            for (iter = 1; iter <= endTime && !converged; ++iter)
            {
                const ParStepResidual r = solver.step();
                if (iter == 1) lap("firstStep");   // first step pays graph capture + first-touch allocation
                if (master && (iter == 1 || iter % 20 == 0))
                    std::printf("  iter %4d:  Ux %.3e  p %.3e  | avg krylov/iter %ld\n",
                                iter, r.Ux, r.p, solver.krylovIters() / iter);
                // residualControl on U/p (the turbulence fields co-converge; step() returns only U/p residuals).
                converged = hasRC && ok(r.p, rcP) && ok(r.Ux, rcU);
            }
            nIter = converged ? iter - 1 : endTime;
            if (master)
                std::printf(converged ? "SIMPLE solution converged in %d iterations\n"
                                      : "SIMPLE reached endTime (%d iterations)\n", nIter);

            Ug = detail::gatherToGlobal(P, solver.U());   // D2H once, at write time
            pg = detail::gatherToGlobal(P, solver.p());
            if (turbulent)
            {
                kg = detail::gatherToGlobal(P, solver.kLocal());
                eg = detail::gatherToGlobal(P, solver.epsLocal());
                ng = detail::gatherToGlobal(P, solver.nutLocal());
            }
        }

        // pRefCell/pRefValue. A CLOSED domain (no fixedValue p patch) has an all-Neumann pressure equation,
        // which is singular: p is determined only up to an additive constant. OF pins it with
        // pEqn.setReference(); we instead solve the consistent singular system and pin the LEVEL here -- the
        // field is identical either way (verified on the 3D cavity: after removing the offset the two agree to
        // 2.8e-07), but without this the written p carries an arbitrary constant and will not match OF.
        {
            bool needRef = true;
            for (const PatchFieldData<scalar>& b : pfd.boundary)
                if (b.type == "fixedValue" || b.type == "totalPressure" || b.type == "outletInlet"
                    || b.type == "fixedMean" || b.type == "prghPressure")
                    needRef = false;
            if (needRef && !pg.empty())
            {
                const label  refCell = static_cast<label>(simple ? simple->intOr("pRefCell", 0) : 0);
                const scalar refVal  = simple ? simple->scalarOr("pRefValue", 0.0) : 0.0;
                if (refCell >= 0 && refCell < static_cast<label>(pg.size()))
                {
                    const scalar shift = refVal - pg[refCell];
                    for (scalar& v : pg) v += shift;
                    if (master)
                        std::printf("  closed domain: p pinned to pRefValue=%g at pRefCell=%ld (shift %.3e)\n",
                                    refVal, (long)refCell, shift);
                }
            }
        }

        if (master)
        {
            const std::string outDir = caseDir + "/" + std::to_string(nIter);
            std::filesystem::create_directories(outDir);
            const std::string src = fieldDir + "/";
            writeVolField(src + "U", outDir + "/U", Ug, gm.patches(), precision);
            writeVolField(src + "p", outDir + "/p", pg, gm.patches(), precision);
            if (turbulent)
            {
                writeVolField(src + "k",       outDir + "/k",       kg, gm.patches(), precision);
                writeVolField(src + (sst ? "omega" : "epsilon"), outDir + "/" + (sst ? "omega" : "epsilon"), eg, gm.patches(), precision);
                writeVolField(src + "nut",     outDir + "/nut",     ng, gm.patches(), precision);
            }
            std::printf("wrote %s\n", outDir.c_str());
        }
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "brae -parallel (rank %d): %s\n", rank, e.what());
        rc = 1;
    }
    Pstream::finalize();
    return rc;
}

} // namespace brae
