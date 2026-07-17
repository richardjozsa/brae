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
// LAMINAR ONLY (Phase 4a). Turbulence is 4b: ParallelDeviceSimple has no k/epsilon transport, so a RAS case
// must be REFUSED here rather than silently solved as laminar -- that would look like it worked and be wrong.
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

#include <cstdio>
#include <sstream>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

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

        // Phase 4a is laminar. Refuse RAS rather than solve it as laminar behind the user's back.
        const std::string simType = turbProps.wordOr("simulationType", "laminar");
        if (simType != "laminar")
            throw std::runtime_error(
                "brae -parallel: simulationType '" + simType + "' is not supported on the multi-GPU device "
                "path yet (Phase 4a is laminar; turbulence is 4b). Use the host solver "
                "`mpirun -np N brae_simpleFoam -case <dir>` for RAS in parallel, or `brae -case <dir>` for "
                "RAS on a single GPU.");

        // fvSchemes div(phi,U). The distributed path implements Gauss upwind, bounded, and linearUpwind (the
        // matrix is upwind either way; linearUpwind adds the deferred gradient correction, cut faces included).
        // Anything else -- limitedLinear, LUST, linearUpwindV, plain Gauss linear -- is a DIFFERENT
        // discretisation, and silently substituting upwind produces a converged, plausible, WRONG answer (on
        // the cavity that was a ~6% field difference). Refuse those, exactly as RAS is refused above.
        bool boundedDiv = false, linUpwind = false;
        {
            std::string divLine;
            std::istringstream fsch(readFileExpanded(caseDir + "/system/fvSchemes"));
            std::string ln;
            while (std::getline(fsch, ln))
                if (ln.find("div(phi,U)") != std::string::npos) divLine = ln;
            if (!divLine.empty())
            {
                boundedDiv = divLine.find("bounded") != std::string::npos;
                linUpwind  = divLine.find("linearUpwind") != std::string::npos;
                // NB "linearUpwind" has a capital U, so it does NOT contain the substring "upwind" -- the two
                // must be tested separately or a linearUpwind case reads as "no upwind scheme at all".
                const bool upwindFamily = (divLine.find("upwind") != std::string::npos) || linUpwind;
                // linearUpwindV adds OF's vector direction limiter on top of linearUpwind -- NOT implemented,
                // and it contains the substring "linearUpwind", so it must be excluded explicitly.
                const bool unsupported = divLine.find("linearUpwindV") != std::string::npos
                                      || divLine.find("limitedLinear") != std::string::npos
                                      || divLine.find("LUST") != std::string::npos;
                if (!upwindFamily || unsupported)
                    throw std::runtime_error(
                        "brae -parallel: div(phi,U) scheme '" + divLine + "' is not implemented on the "
                        "multi-GPU device path (it supports 'Gauss upwind', 'bounded Gauss upwind' and "
                        "'[bounded] Gauss linearUpwind grad(U)'). Substituting a scheme would converge to a "
                        "DIFFERENT answer than fvSchemes asks for, so this is refused rather than solved "
                        "wrongly. Use `brae -case <dir>` (single GPU) for the full scheme set.");
            }
        }

        const FoamDict* rf  = fvSolution.subDict("relaxationFactors");
        const FoamDict* eqs = rf ? rf->subDict("equations") : nullptr;
        const FoamDict* fld = rf ? rf->subDict("fields") : nullptr;
        const scalar relaxU = eqs ? eqs->scalarOr("U", 1.0) : 1.0;
        const scalar relaxP = fld ? fld->scalarOr("p", 1.0) : 1.0;

        const FoamDict* solvers = fvSolution.subDict("solvers");
        auto solverTol = [&](const std::string& f, scalar def)
        {
            const FoamDict* s = solvers ? solvers->subDict(f) : nullptr;
            return s ? s->scalarOr("tolerance", def) : def;
        };
        const scalar tolP = solverTol("p", 1e-6), tolU = solverTol("U", 1e-8);

        const FoamDict* simple = fvSolution.subDict("SIMPLE");
        const FoamDict* resCtl = simple ? simple->subDict("residualControl") : nullptr;
        const bool   hasRC = (resCtl != nullptr);
        const scalar rcP = resCtl ? resCtl->scalarOr("p", -1) : -1;
        const scalar rcU = resCtl ? resCtl->scalarOr("U", -1) : -1;

        PrimitiveMesh gm;
        gm.read(caseDir + "/constant/polyMesh");
        const label nC = gm.nCells();

        // decompose in core: SCOTCH on master, broadcast so every rank agrees on the same partitioning
        std::vector<label> cellToPart(nC, 0);
        if (master) cellToPart = scotchDecompose(gm, nproc);
        Pstream::broadcast(cellToPart.data(), nC, 0);
        const Partition P(gm, cellToPart, rank);

        const FieldData<vector> Ufd = readField<vector>(fieldDir + "/U");
        const FieldData<scalar> pfd = readField<scalar>(fieldDir + "/p");
        GeometricField<vector> U0 = distributeField<vector>(Ufd, gm.patches(), P.Lm, P.lp, P.procW, rank);
        U0.evaluateBoundary();
        GeometricField<scalar> p0 = distributeField<scalar>(pfd, gm.patches(), P.Lm, P.lp, P.procW, rank);
        p0.evaluateBoundary();

        if (master)
        {
            std::printf("brae (device, distributed) | case=%s np=%d | laminar | nu=%.3g | %ld cells\n",
                        caseDir.c_str(), nproc, nu, (long)nC);
            std::printf("  relax U=%.2g p=%.2g | tol p=%.1g U=%.1g | endTime=%d | residualControl=%s | div(phi,U)=%sGauss upwind\n",
                        relaxU, relaxP, tolP, tolU, endTime, hasRC ? "on" : "off",
                        (std::string(boundedDiv ? "bounded " : "") + (linUpwind ? "Gauss linearUpwind" : "Gauss upwind")).c_str());
        }

        std::vector<vector> Ug;
        std::vector<scalar> pg;
        int nIter = 0;
        {   // scope: the solver's symmetric-heap buffers must be freed BEFORE Pstream::finalize
            ParallelDeviceSimple solver(P, U0, p0, nu, relaxU, relaxP, tolU, tolP, 2000, boundedDiv, linUpwind);

            auto ok = [](scalar res, scalar ctl) { return ctl < 0 || res < ctl; };
            bool converged = false;
            int iter = 0;
            for (iter = 1; iter <= endTime && !converged; ++iter)
            {
                const ParStepResidual r = solver.step();
                if (master && (iter == 1 || iter % 50 == 0))
                    std::printf("  iter %4d:  Ux %.3e  p %.3e\n", iter, r.Ux, r.p);
                converged = hasRC && ok(r.p, rcP) && ok(r.Ux, rcU);
            }
            nIter = converged ? iter - 1 : endTime;
            if (master)
                std::printf(converged ? "SIMPLE solution converged in %d iterations\n"
                                      : "SIMPLE reached endTime (%d iterations)\n", nIter);

            Ug = detail::gatherToGlobal(P, solver.U());   // D2H once, at write time
            pg = detail::gatherToGlobal(P, solver.p());
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
