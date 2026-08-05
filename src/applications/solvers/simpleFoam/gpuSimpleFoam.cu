// cf gpuSimpleFoam, steady incompressible SIMPLE solver, single-GPU device-resident. Reads the case dicts
// (controlDict / fvSolution / transportProperties / turbulenceProperties) and start fields, runs the whole
// SIMPLE(+kEpsilon) loop on the GPU via DeviceSimpleSolver (U/p/phi/k/eps/nut never leave the device between
// iterations), and writes the converged fields when fvSolution residualControl is met or controlDict endTime
// is reached. The momentum is the faithful incompressible stress div(phi,U)-laplacian(nuEff,U)-div(nuEff*
// dev2(T(grad U))); the pressure uses the device AMG-PCG. Mirrors the host brae_simpleFoam control flow.
//
//   brae -case <caseDir>
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "foam_field_writer.cuh"
#include "forces.cuh"
#include "mrf_read.cuh"
#include "fv_options.cuh"
#include "turbulent_inlet.cuh"
#include "foam_dict.cuh"
#include "scheme_parse.cuh"
#include "linear_solver_setup.cuh"   // readLinearSolverControls (shared with gpuRhoSimpleFoam)   // parseFvSchemesControls: shared fvSchemes div/laplacian scheme parse (steady + transient)
#include "solver_dispatch.cuh"   // dispatchSolver + execSibling: route to the solver / component that owns the work
#include "benchmark.cuh"         // brae benchmark [sample]: the standard workload, pulled from the template repo
#include "turbulence_setup.cuh"   // readTurbulenceModel + readTurbulenceFields (shared with pimpleFoam)
#include "sweep_cases.cuh"   // brae -cases c1 c2 ...: multi-GPU mesh/parameter study (orchestrator mode)
#include "fvc.cuh"
#include "device_simple_foam.cuh"
#include "coded_bc_setup.cuh"         // CodedBCSpec + parseCodedBCs + setupCodedBCs (shared with gpuPimpleFoam)

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <deque>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>

using namespace brae;

static void printUsage()
{
    std::printf(
"brae, a GPU-native, OpenFOAM-compatible CFD solver. The whole solve runs on one GPU; reads a standard\n"
"OpenFOAM case and writes standard time dirs.\n\n"
"Solvers (picked from the case's controlDict `application`, so `brae` is the only command you type):\n"
"  simpleFoam       steady incompressible, RAS/laminar\n"
"  pimpleFoam       transient incompressible, URANS/DES/LES/laminar\n"
"Any other application stops at start-up rather than run the case with the wrong solver.\n\n"
"Subcommands (the only two reserved words; anything else leading is a case directory):\n"
"  brae benchmark [sample]        run the standard workload, writes brae-benchmark.json\n"
"  brae benchmark --list          the samples published at github.com/simd-ai/brae-bench\n"
"  brae node register|status|unregister\n"
"                                 join this machine to the Brae network (needs brae-agent)\n"
"A case directory called 'node' or 'benchmark' is addressed as `brae -case node`.\n\n"
"Usage:\n"
"  brae [-case <dir>]             solve an OpenFOAM case (default: the current directory)\n"
"  mpirun -np N brae -case <dir> -parallel\n"
"                                 solve across N GPUs, one rank per GPU (laminar only; needs P2P/NVLink).\n"
"                                 Decomposes in core and reconstructs on write -- no decomposePar step.\n"
"  brae -partition [-case <dir>]  build + cache the mesh and AMG hierarchy, then exit (no solve)\n"
"  brae -cases <d1> <d2> ...      run several cases at once, one per GPU (mesh/parameter study)\n"
"  brae --help                    show this message\n\n"
"A case is a standard OpenFOAM directory (0/ constant/ system/, ASCII or binary mesh). No decomposePar\n"
"needed; brae auto-partitions for the GPU. With -cases, extra cases queue as GPUs free up.\n\n"
"Environment:\n"
"  BRAE_GPUS=N        override the detected GPU count (for -cases)\n"
"  BRAE_JOBS=N        how many cases to run at once with -cases (default: number of GPUs)\n"
"  BRAE_PCG_DEVICE=0  disable the device-resident PCG (on by default)\n"
"  BRAE_AMG_FP32=0    use the FP64 AMG preconditioner instead of FP32 (on by default)\n\n"
"Docs and benchmarks: https://github.com/simd-ai/brae\n");
}

int main(int argc, char** argv)
{
    try
    {
        // Subcommands. ONLY these three leading words are reserved -- everything else is a case directory, so
        // `brae myCase` is untouched. A case actually named `node`, `benchmark` or `job` is reached with -case.
        if (argc > 1 && argv[1][0] != '-')
        {
            const std::string word = argv[1];
            if (word == "benchmark")
            {
                std::vector<std::string> rest(argv + 2, argv + argc);
                std::error_code ec;
                const std::filesystem::path self = std::filesystem::read_symlink("/proc/self/exe", ec);
                return bench::runBenchmark(rest, ec ? std::string("brae") : self.string());
            }
            if (word == "node")
            {
                // The node service is a separate, CUDA-free binary; `brae` is only the front door to it.
                std::vector<std::string> args(argv + 1, argv + argc);
                execSibling("brae-agent", args, "node subcommand -> brae-agent", "`brae node` (the node service)");
            }
            if (word == "job")
            {
                // `brae job run`, not `brae run`. A bare `run` was tried and tests/brae_subcommands.sh
                // refused it, correctly: a case directory called `run` is commonplace, and the rule here is
                // that only reserved words are subcommands and everything else is a case. Stealing `run`
                // would have made `brae run` mean different things in different directories.
                std::vector<std::string> args(argv + 1, argv + argc);
                execSibling("brae-agent", args, "job subcommand -> brae-agent", "`brae job` (submitting work)");
            }
        }
        for (int i = 1; i < argc; ++i)
        {
            const std::string h = argv[i];
            if (h == "--help" || h == "-h")
            {
                printUsage();
                return 0;
            }
        }
        // -cases c1 c2 ... : run several cases at once, one per GPU (mesh/parameter study). The parent
        // orchestrates (forks one child `brae -case cX` per GPU, no CUDA here); a plain -case is unaffected.
        for (int i = 1; i < argc; ++i)
            if (std::string(argv[i]) == "-cases")
            {
                std::vector<std::string> sweep;
                for (int j = i + 1; j < argc && argv[j][0] != '-'; ++j)
                    sweep.push_back(argv[j]);
                if (!sweep.empty()) return braesweep::runSweepCases(sweep);
            }
        // -parallel (the OpenFOAM convention): the distributed DEVICE path, one rank per GPU.
        //   mpirun -np N brae -case <dir> -parallel
        // Gated on the flag rather than on Pstream::nProcs() so a plain `brae -case ...` never initialises
        // MPI -- which also keeps the -cases fork orchestrator above clear of it.
        // -parallel (multi-GPU) is OUT OF SCOPE; the distributed solver lives in legacy/ and is not built.
        // REFUSE rather than fall through to the single-GPU path: under `mpirun -np N` that would run N
        // redundant identical solves, each writing over the others' time directories.
        for (int i = 1; i < argc; ++i)
            if (std::string(argv[i]) == "-parallel")
                throw std::runtime_error(
                    "brae: -parallel (multi-GPU) is not supported in this build. The distributed solver was "
                    "moved to legacy/ and is out of scope; see legacy/README.md. Run brae single-GPU without "
                    "-parallel (and without mpirun).");
        std::string caseDir = ".";
        bool partition = false;                              // -partition: build mesh + AMG caches, then exit (no solve)
        for (int i = 1; i < argc; ++i)
        {
            const std::string a = argv[i];
            if (a == "-case" && i + 1 < argc)
                caseDir = argv[++i];
            else if (a == "-partition")
                partition = true;
            else if (a[0] != '-')
                caseDir = a;
        }
        // The case names its solver (controlDict `application`), so `brae` is the only command a user types: this
        // executable keeps the steady cases and hands the others to the brae solver that owns them, before any dict
        // read or CUDA init. -partition is solver-agnostic prep (mesh + AMG cache) and always runs here.
        // Registry + rules: solvers/common/solver_dispatch.cuh.
        if (!partition) dispatchSolver(caseDir, argc, argv);

        // -partition is cf's analogue of OF decomposePar: do the one-time prep (parse mesh + build AMG hierarchy) and
        // persist it to constant/polyMesh/.brae_mesh|amgcache, so the actual run reloads it warm. Forces the cache write.
        if (partition) setenv("BRAE_MESH_CACHE", "1", 1);

        // controls from the case dictionaries
        const FoamDict controlDict = readDict(caseDir + "/system/controlDict");
        const FoamDict fvSolution  = readDict(caseDir + "/system/fvSolution");
        const FoamDict transport   = readDict(caseDir + "/constant/transportProperties");
        const FoamDict turbProps   = readDict(caseDir + "/constant/turbulenceProperties");

        std::string startStr = controlDict.wordOr("startTime", "0");
        // OF startFrom: 'latestTime' restarts from the newest numeric time directory, 'firstTime' from the earliest
        // (default 'startTime' uses startTime). brae previously IGNORED startFrom, so a 'latestTime' restart silently
        // re-ran from startTime (scratch). Resolve it by scanning the case's numeric time dirs that hold a U field.
        {
            const std::string startFrom = controlDict.wordOr("startFrom", "startTime");
            if (startFrom == "latestTime" || startFrom == "firstTime")
            {
                namespace fs = std::filesystem;
                std::error_code ec;
                double best = 0.0; std::string bestName;
                for (const auto& e : fs::directory_iterator(caseDir, ec))
                {
                    if (!e.is_directory()) continue;
                    const std::string nm = e.path().filename().string();
                    char* end = nullptr; const double t = std::strtod(nm.c_str(), &end);
                    if (end == nm.c_str() || *end != '\0') continue;   // not a pure number -> skip constant/system/*.orig
                    if (!(fs::exists(e.path() / "U") || fs::exists(e.path() / "U.gz"))) continue;   // must be a field dir
                    if (bestName.empty() || (startFrom == "latestTime" ? t > best : t < best)) { best = t; bestName = nm; }
                }
                if (!bestName.empty() && bestName != startStr)
                {
                    std::fprintf(stderr, "brae: startFrom %s -> starting from time '%s'\n", startFrom.c_str(), bestName.c_str());
                    startStr = bestName;
                }
            }
        }
        // OF `restore0Dir` convention (NOT a solver fallback): tutorials needing a mesh-prep step (snappyHexMesh /
        // setFields / changeDictionary, e.g. motorBike) ship the flow fields in <startTime>.orig, because
        // snappyHexMesh writes mesh-level fields (cellLevel/pointLevel/...) INTO <startTime>. OpenFOAM's Allrun copies
        // 0.orig -> 0 before solving; cf does exactly the same SETUP step here (only when <startTime> has no U), so the
        // solver then reads <startTime> identically to OF, behaviour unchanged vs OF, no read-from-.orig special case.
        {
            namespace fs = std::filesystem;
            std::error_code ec;
            const std::string d0 = caseDir + "/" + startStr, dOrig = d0 + ".orig";
            const bool haveU = fs::exists(d0 + "/U") || fs::exists(d0 + "/U.gz");
            if (!haveU && (fs::exists(dOrig + "/U") || fs::exists(dOrig + "/U.gz")))
            {
                std::fprintf(stderr, "brae: %s/U not found -> restoring %s/* into %s (OpenFOAM restore0Dir convention)\n",
                             d0.c_str(), dOrig.c_str(), d0.c_str());
                fs::create_directories(d0, ec);
                for (const auto& e : fs::directory_iterator(dOrig, ec))
                    fs::copy(e.path(), d0 + "/" + e.path().filename().string(),
                             fs::copy_options::overwrite_existing | fs::copy_options::recursive, ec);
            }
        }
        const std::string fieldDir = caseDir + "/" + startStr;   // solver reads <startTime> exactly as in OpenFOAM
        const int   endTime   = controlDict.intOr("endTime", 1000);
        // Steady simpleFoam's endTime is an INTEGER iteration count. A fractional endTime (e.g. 0.3, copy-pasted from
        // a transient case) truncates to 0 -> the loop runs 0 iterations and would write the initial field as "the
        // solution". Refuse endTime < 1.
        if (endTime < 1)
            throw std::runtime_error("controlDict endTime = " + std::to_string(endTime) + " (< 1). Steady simpleFoam"
                " runs an integer iteration count; a fractional endTime truncates to 0 and would write the initial"
                " field as the solution. Set endTime to the number of SIMPLE iterations.");
        const int   precision = controlDict.intOr("writePrecision", 16);

        DeviceSimpleControls ctl;
        ctl.caseDir = caseDir;
        ctl.writeCache = std::getenv("BRAE_MESH_CACHE") != nullptr;   // -partition (above) or the env
        ctl.nu = transport.scalarOr("nu", 1e-5);
        // brae is Newtonian-only; a non-Newtonian transportModel has no top-level nu, so it would silently run with
        // the default constant nu (the wrong viscosity for a shear-thinning/thickening fluid). Fail loud instead.
        {
            const std::string tModel = transport.wordOr("transportModel", "Newtonian");
            if (tModel != "Newtonian")
                throw std::runtime_error("constant/transportProperties transportModel '" + tModel + "' is not"
                    " supported -- brae is Newtonian-only and would silently use a constant nu. Only 'Newtonian' is supported.");
        }
        // read schemes: div(phi,U) "bounded" -> -Sp(div(phi),.); "linearUpwind" -> deferred gradient correction;
        // laplacian/snGrad "corrected" -> non-orthogonal correction (nonOrthDeltaCoeffs implicit + corrVec.grad explicit).
        parseFvSchemesControls(caseDir, ctl);
        const std::string simType = turbProps.wordOr("simulationType", "laminar");
        ctl.turbulent = (simType == "RAS");
        if (simType != "RAS" && simType != "laminar")
            throw std::runtime_error("brae: unsupported simulationType '" + simType + "' (RAS or laminar)");
        readTurbulenceModel(turbProps, ctl);

        // Scalar linearUpwind is gated OFF here as a COLD-START STABILITY guard, not for accuracy. The
        // original comment claimed it "degrades turbulence accuracy vs OF"; that was measured with the old
        // line-based fvSchemes parser (the one that leaked schemes between statements) and is wrong. What
        // the re-measurement actually shows:
        //   - discretisation is CORRECT: one iteration from OF's own converged pitzDaily state, tight
        //     solvers, linearUpwind on k -> k agrees with OF to 1.6e-06 (omega 3.3e-05, p exact).
        //   - the compressible duct converges to OF at k 2.0e-06 / nut 8.1e-07 with it honoured, and is
        //     1.6e-02 off on nut with it downgraded -- so rhoSimpleFoam now HONOURS it.
        //   - but from a COLD start pitzDaily SST diverges (NaN ~iter 550) with linearUpwind on k, where
        //     OF converges at 357. Bisected: k's correction diverges, omega's is stable (upwind/upwind and
        //     upwind/linearUpwind both converge). Ruled out: SIMPLEC (diverges with consistent no too),
        //     gradient limiting (both unlimited), regex relaxation (brae honours ".*" 0.9), the Pk
        //     production limiter and k/omega bounding (both present and OF-exact).
        // So the guard stays until that cold-start path is fixed, but it is a robustness workaround with a
        // known accuracy cost, NOT the validated behaviour. BRAE_SCALAR_LINEARUPWIND=1 honours the scheme.
        if (!std::getenv("BRAE_SCALAR_LINEARUPWIND"))
        {
            // Never silently honour-then-ignore: if fvSchemes asked for it, say that upwind is running.
            if (ctl.luK || ctl.luEps)
                std::fprintf(stderr, "brae WARNING: div(phi,<turbulence scalar>) requested 'linearUpwind' but brae is "
                             "running UPWIND (cold-start stability guard; the discretisation itself matches OF to "
                             "1.6e-06). Set BRAE_SCALAR_LINEARUPWIND=1 to honour the requested scheme.\n");
            ctl.luK = false;
            ctl.luEps = false;
        }

        readRelaxationFactors(fvSolution, ctl);   // shared: solvers/common/linear_solver_setup.cuh

        // fvSolution -> ctl, through the SHARED reader in solvers/common/linear_solver_setup.cuh.
        // This used to be an inline copy here; the compressible driver was ported from it and silently
        // dropped fifteen of these controls. One reader means a new driver gets the whole set or none.
        const std::string second = ctl.sst ? "omega" : "epsilon";
        readLinearSolverControls(fvSolution, second, ctl);

        const FoamDict* simple = fvSolution.subDict("SIMPLE");
        const FoamDict* resCtl = simple ? simple->subDict("residualControl") : nullptr;
        const bool hasRC = (resCtl != nullptr);
        const scalar rcP = resCtl ? resCtl->scalarOr("p", -1) : -1, rcU = resCtl ? resCtl->scalarOr("U", -1) : -1;
        // consistent / nNonOrthogonalCorrectors / bodyForce are read by readLinearSolverControls above.

        std::printf("brae (device-resident) | case=%s | %s%s | nu=%.3g\n", caseDir.c_str(),
                    simType.c_str(), ctl.turbulent ? (ctl.sst ? " (kOmegaSST)" : " (kEpsilon)") : "", ctl.nu);
        std::printf("  relax U=%.2g p=%.2g | tol p=%.1g U=%.1g | endTime=%d | residualControl=%s\n",
                    ctl.relaxU, ctl.relaxP, ctl.tolP, ctl.tolU, endTime, hasRC ? "on" : "off");
        std::printf("  schemes: bounded(U=%d,k=%d,eps=%d) linearUpwind(U)=%d nonOrth(corrected)=%d nonOrthLimit=%.3g consistent(SIMPLEC)=%d limitedLinear(k=%d,eps=%d) linearUpwind(k=%d,eps=%d)\n",
                    ctl.bounded, ctl.boundedK, ctl.boundedEps, ctl.linearUpwind, ctl.nonOrth, ctl.nonOrthLimit, ctl.consistent, ctl.limitedK, ctl.limitedEps, ctl.luK, ctl.luEps);
        std::printf("  grad(U) cellLimited k=%.3g (0=unlimited)\n", ctl.gradULimitK);
        std::printf("  linear solver (GaussSeidel from fvSolution; else BiCGStab): U=%d k|nuTilda=%d eps|omega=%d\n", ctl.gsU, ctl.gsK, ctl.gsEps);

        // mesh + start fields (single GPU: read directly, no decomposition)
        const bool timeStartup = std::getenv("BRAE_TIME_STARTUP") != nullptr;
        auto _tsClk = std::chrono::high_resolution_clock::now();
        auto _tsLap = [&](const char* what)
        {
            if (timeStartup)
            {
                auto n = std::chrono::high_resolution_clock::now();
                std::fprintf(stderr, "[startup] %-22s %6.2f s\n", what, std::chrono::duration<double>(n-_tsClk).count());
                _tsClk = n;
            }
        };
        PrimitiveMesh m;
        m.read(caseDir + "/constant/polyMesh");
        _tsLap("mesh read");
        FvGeometry g;
        g.build(m);
        _tsLap("geometry build");
        const std::vector<FvPatch> fvp = buildPatches(m, g);
        const label nC = m.nCells();

        GeometricField<vector> U = buildField<vector>(readField<vector>(fieldDir + "/U"), fvp, nC);
        U.evaluateBoundary();
        GeometricField<scalar> p = buildField<scalar>(readField<scalar>(fieldDir + "/p"), fvp, nC);
        p.evaluateBoundary();
        // pressure needs a reference iff NO p patch fixes the value (singular all-Neumann system, e.g. closed
        // lid-driven cavity / fixedFluxPressure-walled domains). Then adjustPhi + pEqn.setReference (OF needReference).
        ctl.needRef = true;
        // a fixedValue-p OR a freestreamPressure (outletInlet, cat 4: fixedValue at outflow) references the pressure.
        for (const auto& bf : p.boundary)
            if (bf->fixesValue() || bf->bcCategory() == 4)
            {
                ctl.needRef = false;
                break;
            }
        if (ctl.needRef)
        {
            ctl.pRefCell  = simple ? (label)simple->intOr("pRefCell", 0) : 0;
            ctl.pRefValue = simple ? simple->scalarOr("pRefValue", 0.0) : 0.0;
            std::printf("  pressure needs reference (no fixedValue-p): pRefCell=%d pRefValue=%.4g\n",
                        (int)ctl.pRefCell, ctl.pRefValue);
        }
        SurfaceScalarField phi = fvc::flux(U, m, g, fvp);

        const std::string secondName = ctl.sst ? "omega" : "epsilon";   // the 2nd turbulence scalar
        TurbulenceFields tf = readTurbulenceFields(fieldDir, fvp, nC, ctl, secondName, U);

        // MRF rotating zone (constant/MRFProperties + polyMesh/cellZones), if present
        const MRFConfig mrfCfg = readMRFProperties(caseDir + "/constant");
        MRFZone mrfZone;
        if (mrfCfg.active)
        {
            const auto zones = readCellZones(caseDir + "/constant/polyMesh");
            const auto it = zones.find(mrfCfg.cellZone);
            std::vector<label> zoneCells = (it != zones.end()) ? it->second : std::vector<label>{};
            if (zoneCells.empty())
            {
                // A named zone that isn't in cellZones is almost always a typo (or a binary/unparsed zone): the
                // old silent whole-mesh fallback then turned the ENTIRE domain into a rotating frame. Refuse that,
                // and report the zones that ARE present -- EXCEPT for the explicit 'all' convention (deliberate
                // whole-domain rotation, e.g. rotatingCylinders), which we honour but announce loudly.
                if (mrfCfg.cellZone == "all")
                {
                    std::printf("  MRF: cellZone 'all' -> rotating the WHOLE mesh (%d cells)\n", nC);
                    zoneCells.resize(nC);
                    for (label c = 0; c < nC; ++c) zoneCells[c] = c;
                }
                else
                {
                    std::string avail;
                    for (const auto& z : zones) avail += (avail.empty() ? "" : ", ") + z.first;
                    throw std::runtime_error(
                        std::string("MRF cellZone '") + mrfCfg.cellZone
                        + "' not found or empty in constant/polyMesh/cellZones (available: "
                        + (avail.empty() ? std::string("<none>") : avail)
                        + "). Refusing to silently rotate the ENTIRE domain -- fix the 'cellZone' name in"
                          " constant/MRFProperties (use 'all' for deliberate whole-mesh rotation).");
                }
            }
            mrfZone = buildMRFZone(m, zoneCells, mrfCfg.axis, mrfCfg.omega, mrfCfg.origin);
            mrfCorrectBoundaryVelocity(U, mrfZone, g, fvp, mrfCfg.nonRotatingPatches);   // in-zone walls -> Omega x r
            phi = fvc::flux(U, m, g, fvp);                                               // re-flux (absolute; setMRF makes it relative)
            std::printf("  MRF: cellZone=%s omega=%.4g axis=(%.2g %.2g %.2g) origin=(%.2g %.2g %.2g) nonRotating=%zu zoneCells=%zu\n",
                        mrfCfg.cellZone.c_str(), mrfCfg.omega, mrfCfg.axis.x, mrfCfg.axis.y, mrfCfg.axis.z,
                        mrfCfg.origin.x, mrfCfg.origin.y, mrfCfg.origin.z, mrfCfg.nonRotatingPatches.size(), zoneCells.size());
        }

        // device-resident SIMPLE loop
        _tsLap("fields + patches");
        DeviceSimpleSolver solver(m, g, fvp, U, p, phi, ctl,
                                  ctl.turbulent ? &tf.k : nullptr, (ctl.turbulent && !ctl.sa) ? &tf.eps : nullptr, ctl.turbulent ? &tf.nut : nullptr,
                                  ctl.lm ? &tf.ReThetat : nullptr, ctl.lm ? &tf.gammaInt : nullptr);
        _tsLap("solver ctor (incl AMG)");
        if (partition)   // caches written by the mesh read + the AMG build above; done, like decomposePar finishing.
        {
            std::printf("brae -partition: mesh + AMG hierarchy cached to %s/constant/polyMesh/ (.brae_meshcache + .brae_amgcache).\n"
                        "  Run the solve normally; it will reload them warm.\n", caseDir.c_str());
            return 0;
        }
        if (mrfCfg.active) solver.setMRF(mrfZone, m, g, mrfCfg.nonRotatingPatches);

        // codedFixedValue / codedMixed on U / p / turbulence scalars -> NVRTC device kernel per coded patch (compiled
        // once; applied each SIMPLE iteration in the momentum predictor). Steady: the coded `t` stays 0 (position-based
        // profiles); a codedFixedValue set here overrides its seed `value`. Fully device-resident.
        setupCodedBCs(solver, fieldDir, fvp, ctl, secondName, "gpuSimpleFoam");

        // fvOptions (system/fvOptions or constant/fvOptions), if present (else an empty no-op list)
        // Read cellZones only when an fvOptions file exists (avoids touching cellZones on cases that have none,
        // and a binary cellZones, which readCellZones does not parse).
        std::map<std::string, std::vector<label>> fvoZones;
        {
            std::ifstream a(caseDir + "/system/fvOptions"), b(caseDir + "/constant/fvOptions");
            if (a.good() || b.good()) fvoZones = readCellZones(caseDir + "/constant/polyMesh");
        }
        const FvOptionsData fvo = readFvOptions(caseDir, fvoZones, g.V(), nC, g.C());
        if (!fvo.unsupported.empty())   // fail loud rather than run a valid-looking case with a silently-dropped source
        {
            std::string msg = "fvOptions contains source(s) brae cannot apply (they would be SILENTLY dropped -> wrong physics):";
            for (const auto& u : fvo.unsupported) msg += "\n  - " + u;
            msg += "\nRemove/disable them, or use a supported form. Supported: vectorSemiImplicitSource,"
                   " explicitPorositySource[DarcyForchheimer], meanVelocityForce, limitVelocity,"
                   " actuationDiskSource[Froude], rotorDisk, velocityDampingConstraint; selectionMode all|cellZone.";
            throw std::runtime_error(msg);
        }
        if (!fvo.empty())
        {
            solver.setFvOptions(fvo);
            if (fvo.rotor.active)   // build the BEM rotor geometry from the mesh (cell centres + face areas) and hand it over
                solver.setRotorDisk(buildDeviceRotorDisk(fvo.rotor, g.C(), g.Sf(), m.owner(), m.neighbour(), m.nInternalFaces()));
            std::printf("  fvOptions: %d source(s)%s%s%s%s%s%s%s%s\n", fvo.count, fvo.hasMomentum ? " momentum" : "",
                        fvo.porActive ? " DarcyForchheimer-porosity" : "", fvo.mvfActive ? " meanVelocityForce" : "",
                        fvo.limUActive ? " limitVelocity" : "", fvo.adActive ? " actuationDiskSource" : "",
                        fvo.rotor.active ? " rotorDiskSource" : "", fvo.vdcActive ? " velocityDampingConstraint" : "",
                        (!fvo.scaSu.empty() || !fvo.scaSp.empty()) ? " scalar(WARN: turbulence-field sources read but not yet applied per-model)" : "");
        }
        else
        {
            std::printf("  No finite volume options present\n");   // OF createFvOptions.H message
        }
        // OF simpleControl::criteriaSatisfied: an unlisted field is not a criterion, and a run only
        // converges if at least one criterion was ACTUALLY checked (see solvers/common/residual_control.cuh).
        int rcChecked = 0;
        auto ok = [&](scalar res, scalar ctlv) { if (ctlv < 0) return true; ++rcChecked; return res < ctlv; };
        // OF controlDict write cadence: writeControl / writeInterval / purgeWrite (ported from Foam::Time)
        const std::string writeControl = controlDict.wordOr("writeControl", "timeStep");
        const scalar writeInterval = controlDict.scalarOr("writeInterval", 1e30);   // OF default GREAT -> only the final state
        const int    purgeWrite    = std::max(0, controlDict.intOr("purgeWrite", 0));
        const scalar deltaT        = controlDict.scalarOr("deltaT", 1.0);
        const scalar startTimeVal  = controlDict.scalarOr("startTime", 0.0);
        long writeTimeIndex = 0;
        auto timeName = [](scalar t) -> std::string   // integer name for whole times (deltaT=1), else %g
        {
            if (t == std::floor(t) && std::fabs((double)t) < 1e15) return std::to_string((long long)std::llround((double)t));
            char b[64];
            std::snprintf(b, sizeof b, "%g", (double)t);
            return std::string(b);
        };
        auto isWriteTime = [&](int it, scalar tval) -> bool   // Foam::Time::operator++ write switch
        {
            if (writeControl == "timeStep")
                return writeInterval >= 1 && (it % (long)writeInterval) == 0;       // Time.C: !(timeIndex % writeInterval)
            if (writeControl == "runTime" || writeControl == "adjustable" || writeControl == "adjustableRunTime")
            {
                const long wi = (long)(((tval - startTimeVal) + 0.5 * deltaT) / writeInterval);
                if (wi > writeTimeIndex)
                {
                    writeTimeIndex = wi;
                    return true;
                }
                return false;
            }
            return false;   // none / clockTime / cpuTime: only the final state (written after the loop)
        };
        std::deque<std::string> writtenTimes;                                       // purgeWrite FIFO (keep the last N)
        const std::string wsrc = fieldDir + "/";
        auto writeTimeDir = [&](const std::string& tname)   // reconstruct + write one time directory
        {
            const std::string outDir = caseDir + "/" + tname;
            std::filesystem::create_directories(outDir);
            // The written fields keep the source field's `#include "include/..."` directives (resolved relative to the
            // time dir). Copy the startTime include/ dir alongside so OpenFOAM readers (paraFoam, postProcess, foamToVTK)
            // resolve them, otherwise the field points at a nonexistent <time>/include/ and fails to load.
            {
                std::error_code ec2;
                if (std::filesystem::exists(fieldDir + "/include"))
                    std::filesystem::copy(fieldDir + "/include", outDir + "/include",
                        std::filesystem::copy_options::recursive | std::filesystem::copy_options::overwrite_existing, ec2);
            }
            writeVolField(wsrc + "U", outDir + "/U", solver.U(), fvp, precision);
            writeVolField(wsrc + "p", outDir + "/p", solver.p(), fvp, precision);
            if (ctl.sa)   // one-equation: the k slot holds nuTilda
            {
                writeVolField(wsrc + "nuTilda", outDir + "/nuTilda", solver.k(),   fvp, precision);
                writeVolField(wsrc + "nut",     outDir + "/nut",     solver.nut(), fvp, precision);
            }
            else if (ctl.turbulent)
            {
                writeVolField(wsrc + "k", outDir + "/k", solver.k(), fvp, precision);
                writeVolField(wsrc + secondName, outDir + "/" + secondName, solver.eps(), fvp, precision);
                writeVolField(wsrc + "nut", outDir + "/nut", solver.nut(), fvp, precision);
                if (ctl.lm)   // kOmegaSSTLM transition fields
                {
                    writeVolField(wsrc + "ReThetat", outDir + "/ReThetat", solver.ReThetat(), fvp, precision);
                    writeVolField(wsrc + "gammaInt", outDir + "/gammaInt", solver.gammaInt(), fvp, precision);
                }
            }
            std::printf("written %s/{U,p%s}\n", outDir.c_str(),
                        ctl.turbulent ? (ctl.sa ? ",nuTilda,nut" : ctl.sst ? ",k,omega,nut" : ",k,epsilon,nut") : "");
            if (purgeWrite > 0)   // Foam::Time TimeIO: keep only the last purgeWrite dirs
            {
                if (writtenTimes.empty() || writtenTimes.back() != tname) writtenTimes.push_back(tname);
                while ((int)writtenTimes.size() > purgeWrite)
                {
                    std::error_code pec;
                    std::filesystem::remove_all(caseDir + "/" + writtenTimes.front(), pec);
                    writtenTimes.pop_front();
                }
            }
        };

        int iter = 0;
        bool converged = false;
        const auto _runStart = std::chrono::high_resolution_clock::now();          // for OpenFOAM-style ExecutionTime
        double _cumCont = 0.0;                                                     // cumulative continuity error (OF continuityErrs.H)
        for (iter = 1; iter <= endTime && !converged; ++iter)
        {
            const DeviceSimpleResidual r = solver.step();
            {
                // OpenFOAM-style per-iteration report: Time, per-field solver residuals, continuity, turbulence, ExecutionTime.
                const double _et = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - _runStart).count();
                const double cl = (double)deltaT * r.contLocal, cg = (double)deltaT * r.contGlobal;
                _cumCont += cg;
                std::printf("Time = %d\n\n"
                            "smoothSolver:  Solving for Ux, Initial residual = %g, Final residual = %g, No Iterations %d\n"
                            "smoothSolver:  Solving for Uy, Initial residual = %g, Final residual = %g, No Iterations %d\n"
                            "smoothSolver:  Solving for Uz, Initial residual = %g, Final residual = %g, No Iterations %d\n"
                            "GAMG:  Solving for p, Initial residual = %g, Final residual = %g, No Iterations %d\n"
                            "time step continuity errors : sum local = %g, global = %g, cumulative = %g\n",
                            iter, r.Ux, r.UxFinal, r.UxIters, r.Uy, r.UyFinal, r.UyIters, r.Uz, r.UzFinal, r.UzIters,
                            r.p, r.pFinal, r.pIters, cl, cg, _cumCont);
                for (const auto& e : turbulenceReport())   // Solving for omega/k/epsilon/... in solve order, like OF
                    std::printf("smoothSolver:  Solving for %s, Initial residual = %g, Final residual = %g, No Iterations %d\n",
                                e.field.c_str(), e.perf.initialResidual, e.perf.finalResidual, e.perf.nIterations);
                std::printf("ExecutionTime = %.2f s  ClockTime = %.0f s\n\n", _et, _et);
            }
            // NaN/divergence guard: a non-finite momentum/pressure residual means the solve blew up (FP32 overflow,
            // singular pressure, turbulence blow-up, or an under-stabilised case). Without this the loop runs to
            // endTime (ok(NaN,tol) is always false -> never "converges") and WRITES the NaN field as the solution.
            // Abort loudly and write nothing. Opt out (e.g. to inspect the field) with BRAE_ALLOW_NONFINITE=1.
            if (!std::getenv("BRAE_ALLOW_NONFINITE")
                && !(std::isfinite(r.p) && std::isfinite(r.Ux) && std::isfinite(r.Uy) && std::isfinite(r.Uz)))
                throw std::runtime_error(
                    "solution diverged: non-finite residual at iteration " + std::to_string(iter)
                    + " (p=" + std::to_string(r.p) + " Ux=" + std::to_string(r.Ux)
                    + " Uy=" + std::to_string(r.Uy) + " Uz=" + std::to_string(r.Uz) + "). Likely causes:"
                    + " too-loose relaxation, a high-non-orthogonality mesh, a singular pressure system, or"
                    + " turbulence blow-up. No field written. Set BRAE_ALLOW_NONFINITE=1 to continue anyway.");
            // OF residualControl: also gate on every turbulence field (k/epsilon/omega/nuTilda) that lists a target.
            // Previously ONLY p and Ux were checked, so a turbulent case could report "converged" with k/epsilon
            // still far from tol -- the substantive bug this fixes. Unlisted fields have target -1 -> ok() ignores
            // them (OF). U stays gated on Ux alone: brae tracks no valid/solved directions, so the out-of-plane
            // component of a 2D/empty or wedge case has a DEGENERATE residual (stuck ~0.1, never reaching tol) that
            // would wrongly block convergence on every 2D case -- gating all U components needs that infra first.
            rcChecked = 0;
            converged = hasRC && ok(r.p, rcP) && ok(r.Ux, rcU);
            if (converged)
                for (const auto& e : turbulenceReport())
                    if (!ok(e.perf.initialResidual, resCtl->scalarOr(e.field, -1))) { converged = false; break; }
            // OF's `checked` safety: `residualControl { }`, or a dict naming only fields brae does not
            // check, must NOT report convergence. Without this brae stopped after ONE iteration and wrote
            // a plausible-looking field set (simpleControl.C:51-57).
            converged = converged && rcChecked > 0;
            const scalar tval = startTimeVal + (scalar)iter * deltaT;               // OF time value at this step
            if (!converged && iter != endTime && isWriteTime(iter, tval)) writeTimeDir(timeName(tval));  // intermediate writes
        }
        const int nIter = converged ? iter - 1 : endTime;
        std::printf(converged ? "SIMPLE solution converged in %d iterations\n"
                              : "SIMPLE reached endTime (%d iterations)\n", nIter);

        // BRAE_DUMP_PHI: write the conservative face flux, the one quantity that carries between SIMPLE
        // iterations and is not in any written cell field. Diagnostic only.
        if (std::getenv("BRAE_DUMP_PHI"))
        {
            const std::vector<scalar> phiI = solver.phiInternal();
            FILE* fp = std::fopen(std::getenv("BRAE_DUMP_PHI"), "w");
            if (fp)
            {
                for (std::size_t i = 0; i < phiI.size(); ++i) std::fprintf(fp, "%.17g\n", phiI[i]);
                std::fclose(fp);
            }
        }
        // BRAE_DUMP_CONTINUITY: localise the per-cell continuity imbalance R[c]=sum_f phi_f and bucket sum|R| by
        // region (wall-adjacent / farfield-adjacent / interior) to find WHERE continuity fails to close.
        if (std::getenv("BRAE_DUMP_CONTINUITY"))
        {
            const std::vector<scalar> phiI = solver.phiInternal(), phiB = solver.phiBoundary();
            std::vector<scalar> R(nC, 0.0);
            for (label f = 0; f < m.nInternalFaces(); ++f)
            {
                R[m.owner()[f]] += phiI[f];
                R[m.neighbour()[f]] -= phiI[f];
            }
            {
                label bi = 0;
                for (const auto& pp : fvp)
                    for (label i = 0; i < pp.size; ++i)
                        R[pp.faceCells[i]] += phiB[bi++];
            }
            std::vector<int> tag(nC, 0);   // 0=interior 1=wall-adjacent 2=farfield(patch)-adjacent
            for (const auto& pp : fvp)
            {
                const int t = (pp.type == "wall") ? 1 : (pp.type == "patch" ? 2 : 0);
                if (t)
                    for (label i = 0; i < pp.size; ++i)
                        tag[pp.faceCells[i]] = std::max(tag[pp.faceCells[i]], t);
            }
            double sAll = 0, sW = 0, sF = 0, sI = 0, mx = 0;
            label mxc = 0;
            for (label c = 0; c < nC; ++c)
            {
                const double a = std::fabs(R[c]);
                sAll += a;
                if (tag[c] == 1) sW += a;
                else if (tag[c] == 2) sF += a;
                else sI += a;
                if (a > mx) { mx = a; mxc = c; }
            }
            const double inv = sAll > 0 ? 100.0 / sAll : 0.0;
            std::printf("CONTINUITY: sum|div phi|=%.4e  max=%.4e @cell %d  | region share: wall=%.1f%% farfield=%.1f%% interior=%.1f%%\n",
                        sAll, mx, (int)mxc, sW * inv, sF * inv, sI * inv);
            std::vector<label> idx(nC);
            for (label c = 0; c < nC; ++c)
                idx[c] = c;
            const label topN = std::min<label>(12, nC);
            std::partial_sort(idx.begin(), idx.begin() + topN, idx.end(), [&](label a, label b){ return std::fabs(R[a]) > std::fabs(R[b]); });
            std::printf("  top cells (|imbalance|, tag, centroid):\n");
            for (label q = 0; q < topN; ++q)
            {
                const label c = idx[q];
                const vector cc = g.C()[c];
                std::printf("    cell %6d  |R|=%.3e  tag=%d  C=(%.4f %.4f %.4f)\n", (int)c, std::fabs(R[c]), tag[c], cc.x, cc.y, cc.z);
            }
        }

        if (std::getenv("BRAE_DUMP_Y") && ctl.turbulent)   // cell wall distance stats vs OF wallDist (SA destruction ~ 1/y^2)
        {
            const std::vector<scalar> yv = solver.cellY();
            if (!yv.empty())
            {
                double mn = 1e300, mx = 0, sm = 0;
                for (scalar v : yv)
                {
                    mn = std::min(mn, (double)v);
                    mx = std::max(mx, (double)v);
                    sm += v;
                }
                std::printf("cellY: min=%.4e max=%.4e mean=%.4e (n=%zu)\n", mn, mx, sm / yv.size(), yv.size());
            }
        }

        // always write the final (converged / endTime) state; matches OF's writeAndEnd, and feeds purgeWrite
        writeTimeDir(timeName(startTimeVal + (scalar)nIter * deltaT));
        const std::vector<vector> Ug = solver.U();   // converged fields, reused by the force calculation below
        const std::vector<scalar> pg = solver.p();

        // forces on wall patches (OF functionObjects::forces; kinematic, rhoInf=1). Validated vs OF (ctest forces).
        if (ctl.turbulent)
        {
            std::vector<std::string> walls;
            for (const auto& q : fvp)
                if (q.type == "wall") walls.push_back(q.name);
            if (!walls.empty())
            {
                for (label c = 0; c < nC; ++c)
                    U.internal[c] = Ug[c];   // converged fields -> evaluate wall BCs
                p.internal = pg;
                U.evaluateBoundary();
                p.evaluateBoundary();
                const scalar wCmu   = ctl.sst ? ctl.ksstCoeffs.betaStar : ctl.keCoeffs.Cmu;
                const scalar wKappa = ctl.sst ? ctl.ksstCoeffs.kappa    : ctl.keCoeffs.kappa;
                const scalar wE     = ctl.sst ? ctl.ksstCoeffs.E        : ctl.keCoeffs.E;
                // Velocity-based wall nut (SA-Spalding, or nutUSpalding/nutUBlended on kEps/kOmegaSST): use the device
                // wall nut for the viscous force, not the k-based one, matching the BC. nutk cases keep nwb=nullptr.
                const bool velNutWall = ctl.sa || ctl.nutWall != NutWall::Nutk;
                const std::vector<scalar> saNutWall = velNutWall ? solver.nutWall() : std::vector<scalar>();
                const std::vector<scalar>* nwb = velNutWall ? &saNutWall : nullptr;
                const ForceResult F = wallForces(U, p, solver.k(), ctl.nu, m, g, fvp, walls, 1.0, 0.0, vector{0,0,0}, wCmu, wKappa, wE, nwb);
                std::printf("forces (walls, rhoInf=1):  pressure=(%.5e %.5e %.5e)  viscous=(%.5e %.5e %.5e)  total=(%.5e %.5e %.5e)\n",
                            F.pressure.x, F.pressure.y, F.pressure.z, F.viscous.x, F.viscous.y, F.viscous.z, F.total().x, F.total().y, F.total().z);

                // forceCoeffs (Cd/Cl/Cm) if the case defines a forceCoeffs functionObject (controlDict.functions).
                const FoamDict* funcs = controlDict.subDict("functions");
                const FoamDict* fcd = nullptr;
                if (funcs)
                    for (const auto& s : funcs->subs)
                        if (s.second.wordOr("type", "") == "forceCoeffs")
                        {
                            fcd = &s.second;
                            break;
                        }
                if (fcd)
                {
                    auto toV = [](const std::vector<scalar>& a, vector d){ return a.size() >= 3 ? vector{a[0],a[1],a[2]} : d; };
                    const std::vector<std::string> fcP = fcd->wordListOr("patches", walls);
                    const scalar rhoInf = fcd->scalarOr("rhoInf", 1.0), magUInf = fcd->scalarOr("magUInf", 1.0);
                    const scalar Aref = fcd->scalarOr("Aref", 1.0), lRef = fcd->scalarOr("lRef", 1.0);
                    const vector liftDir = toV(fcd->scalarListOr("liftDir", {}), vector{0,1,0});
                    const vector dragDir = toV(fcd->scalarListOr("dragDir", {}), vector{1,0,0});
                    const vector pitchAxis = toV(fcd->scalarListOr("pitchAxis", {}), vector{0,0,1});
                    const vector CofR = toV(fcd->scalarListOr("CofR", {}), vector{0,0,0});
                    const ForceResult Fc = wallForces(U, p, solver.k(), ctl.nu, m, g, fvp, fcP, rhoInf, 0.0, CofR, wCmu, wKappa, wE, nwb);
                    const ForceCoeffs cc = forceCoeffs(Fc, dragDir, liftDir, pitchAxis, rhoInf, magUInf, Aref, lRef);
                    std::printf("forceCoeffs (rhoInf=%.3g magUInf=%.3g Aref=%.3g lRef=%.3g):  Cd=%.6e  Cl=%.6e  Cm=%.6e\n",
                                rhoInf, magUInf, Aref, lRef, cc.Cd, cc.Cl, cc.Cm);
                }
            }
        }
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "brae ERROR: %s\n", e.what());
        return 1;
    }
    return 0;
}
