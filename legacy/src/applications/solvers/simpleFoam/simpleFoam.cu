// cf simpleFoam, steady incompressible SIMPLE solver, MPI-parallel. Mirrors OpenFOAM's simpleFoam:
// reads the case dicts (controlDict / fvSolution / transportProperties / turbulenceProperties), runs
// the SIMPLE loop (momentum + pressure, and kEpsilon::correct when turbulence is RAS) until the
// fvSolution residualControl is met or controlDict endTime is reached, then writes the converged
// fields to the time directory.
//
//   mpirun -np N brae_simpleFoam -case <caseDir>
//
// The domain is decomposed in core (SCOTCH) at startup, no separate decomposePar/reconstructPar step;
// each rank reads the global case and works on its partition, and the result is gathered on write.
#include "primitive_mesh.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "foam_field_writer.cuh"
#include "foam_dict.cuh"
#include "fvc.cuh"
#include "scotch_decomposition.cuh"
#include "field_distribute.cuh"
#include "parallel_simple.cuh"
#include "parallel_kepsilon.cuh"
#include "cf_pstream.cuh"

#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>

using namespace brae;

// Gather a partitioned internal field to the global cell ordering (disjoint cells -> sum reduce).
static std::vector<scalar> gather(const Partition& P, const std::vector<scalar>& loc)
{
    std::vector<scalar> g(P.globalNCells, 0.0);
    for (label c = 0; c < P.nCells(); ++c)
        g[P.Lm.cellProcAddr[c]] = loc[c];
    Pstream::allReduce(g.data(), (int)g.size(), ReduceOp::Sum);
    return g;
}

static std::vector<vector> gather(const Partition& P, const std::vector<vector>& loc)
{
    std::vector<scalar> g(3 * P.globalNCells, 0.0);
    for (label c = 0; c < P.nCells(); ++c)
    {
        const label gc = P.Lm.cellProcAddr[c];
        g[3*gc]=loc[c].x;
        g[3*gc+1]=loc[c].y;
        g[3*gc+2]=loc[c].z;
    }
    Pstream::allReduce(g.data(), (int)g.size(), ReduceOp::Sum);
    std::vector<vector> out(P.globalNCells);
    for (label c = 0; c < P.globalNCells; ++c)
        out[c] = {g[3*c], g[3*c+1], g[3*c+2]};
    return out;
}

int main(int argc, char** argv)
{
    Pstream::init(argc, argv);
    const bool master = Pstream::master();
    const int nproc = Pstream::nProcs(), rank = Pstream::myProcNo();
    try
    {
        std::string caseDir = ".";
        for (int i = 1; i < argc; ++i)
        {
            const std::string a = argv[i];
            if (a == "-case" && i + 1 < argc)
                caseDir = argv[++i];
            else if (a[0] != '-')
                caseDir = a;
        }

        // controls from the case dictionaries
        const FoamDict controlDict = readDict(caseDir + "/system/controlDict");
        const FoamDict fvSolution  = readDict(caseDir + "/system/fvSolution");
        const FoamDict transport   = readDict(caseDir + "/constant/transportProperties");
        const FoamDict turbProps   = readDict(caseDir + "/constant/turbulenceProperties");
        // "bounded Gauss upwind" for div(phi,U)? -> add the -Sp(div(phi),.) term to momentum + k/eps (OF scheme).
        bool boundedDiv = false;
        {
            std::istringstream fsch(readFileExpanded(caseDir + "/system/fvSchemes"));   // $var expanded
            std::string ln;
            while (std::getline(fsch, ln))
                if (ln.find("div(phi,U)") != std::string::npos && ln.find("bounded") != std::string::npos)
                    boundedDiv = true;
        }

        const std::string startStr = controlDict.wordOr("startTime", "0");
        const int   endTime   = controlDict.intOr("endTime", 1000);
        const int   precision = controlDict.intOr("writePrecision", 16);
        const scalar nu       = transport.scalarOr("nu", 1e-5);

        const std::string simType = turbProps.wordOr("simulationType", "laminar");
        bool turbulent = (simType == "RAS");
        if (simType != "RAS" && simType != "laminar")
            throw std::runtime_error("brae_simpleFoam: unsupported simulationType '" + simType + "' (RAS or laminar)");
        KEpsilonCoeffs keCoeffs;
        if (turbulent)
        {
            const FoamDict* ras = turbProps.subDict("RAS");
            const std::string model = ras ? ras->wordOr("RASModel", "") : "";
            if (model != "kEpsilon")
                throw std::runtime_error("brae_simpleFoam: unsupported RASModel '" + model + "' (only kEpsilon)");
            // model coefficients: OF turbulenceProperties RAS.kEpsilonCoeffs (optional; absent keys keep OF defaults).
            KEpsilonCoeffs& c = keCoeffs;
            const FoamDict* kec = ras ? ras->subDict("kEpsilonCoeffs") : nullptr;
            if (kec)
            {
                c.Cmu = kec->scalarOr("Cmu", c.Cmu);
                c.C1 = kec->scalarOr("C1", c.C1);
                c.C2 = kec->scalarOr("C2", c.C2);
                c.C3 = kec->scalarOr("C3", c.C3);
                c.sigmaK = kec->scalarOr("sigmak", c.sigmaK);
                c.sigmaEps = kec->scalarOr("sigmaEps", c.sigmaEps);
                c.kappa = kec->scalarOr("kappa", c.kappa);
                c.E = kec->scalarOr("E", c.E);
            }
            std::printf("  kEpsilonCoeffs%s: Cmu=%.4g C1=%.4g C2=%.4g C3=%.4g sigmak=%.4g sigmaEps=%.4g kappa=%.4g E=%.4g\n",
                        kec ? " (from dict)" : " (OF defaults)", c.Cmu, c.C1, c.C2, c.C3, c.sigmaK, c.sigmaEps, c.kappa, c.E);
        }

        // relaxation factors (OF default 1.0 = no under-relaxation when unspecified).
        const FoamDict* rf  = fvSolution.subDict("relaxationFactors");
        const FoamDict* eqs = rf ? rf->subDict("equations") : nullptr;
        const FoamDict* fld = rf ? rf->subDict("fields") : nullptr;
        const scalar relaxU   = eqs ? eqs->scalarOr("U", 1.0) : 1.0;
        const scalar relaxK   = eqs ? eqs->scalarOr("k", 1.0) : 1.0;
        const scalar relaxEps = eqs ? eqs->scalarOr("epsilon", 1.0) : 1.0;
        const scalar relaxP   = fld ? fld->scalarOr("p", 1.0) : 1.0;

        // linear-solver tolerances (per equation).
        const FoamDict* solvers = fvSolution.subDict("solvers");
        auto solverTol = [&](const std::string& f, scalar def)
        {
            const FoamDict* s = solvers ? solvers->subDict(f) : nullptr;
            return s ? s->scalarOr("tolerance", def) : def;
        };
        const scalar tolP = solverTol("p", 1e-6), tolU = solverTol("U", 1e-8);
        const scalar tolK = solverTol("k", 1e-8), tolE = solverTol("epsilon", 1e-8);

        // residualControl (SIMPLE convergence). -1 = not listed -> not checked.
        const FoamDict* simple = fvSolution.subDict("SIMPLE");
        const FoamDict* resCtl = simple ? simple->subDict("residualControl") : nullptr;
        const bool hasRC = (resCtl != nullptr);
        const scalar rcP = resCtl ? resCtl->scalarOr("p", -1) : -1, rcU = resCtl ? resCtl->scalarOr("U", -1) : -1;
        const scalar rcK = resCtl ? resCtl->scalarOr("k", -1) : -1, rcE = resCtl ? resCtl->scalarOr("epsilon", -1) : -1;

        if (master)
        {
            std::printf("brae_simpleFoam | case=%s np=%d | %s%s | nu=%.3g\n", caseDir.c_str(), nproc,
                        simType.c_str(), turbulent ? " (kEpsilon)" : "", nu);
            std::printf("  relax U=%.2g p=%.2g%s | tol p=%.1g U=%.1g | endTime=%d | residualControl=%s\n",
                        relaxU, relaxP, turbulent ? (" k=" + std::to_string(relaxK)).c_str() : "",
                        tolP, tolU, endTime, hasRC ? "on" : "off");
        }

        // mesh + start fields, decompose, distribute
        PrimitiveMesh gm;
        gm.read(caseDir + "/constant/polyMesh");
        const label nC = gm.nCells();
        const FieldData<vector> Ufd = readField<vector>(caseDir + "/" + startStr + "/U");
        const FieldData<scalar> pfd = readField<scalar>(caseDir + "/" + startStr + "/p");

        std::vector<label> cellToPart(nC, 0);
        if (master) cellToPart = scotchDecompose(gm, nproc);
        Pstream::broadcast(cellToPart.data(), nC, 0);
        Partition P(gm, cellToPart, rank);

        GeometricField<vector> U = distributeField<vector>(Ufd, gm.patches(), P.Lm, P.lp, P.procW, rank);
        U.evaluateBoundary();
        GeometricField<scalar> p = distributeField<scalar>(pfd, gm.patches(), P.Lm, P.lp, P.procW, rank);
        p.evaluateBoundary();
        SurfaceScalarField phi = fvc::flux(U, P.Lm.mesh, P.lg, P.lp);

        GeometricField<scalar> k, eps, nut;
        FieldData<scalar> kfd, efd, nfd;
        if (turbulent)
        {
            kfd = readField<scalar>(caseDir + "/" + startStr + "/k");
            efd = readField<scalar>(caseDir + "/" + startStr + "/epsilon");
            nfd = readField<scalar>(caseDir + "/" + startStr + "/nut");
            k   = distributeField<scalar>(kfd, gm.patches(), P.Lm, P.lp, P.procW, rank);
            k.evaluateBoundary();
            eps = distributeField<scalar>(efd, gm.patches(), P.Lm, P.lp, P.procW, rank);
            eps.evaluateBoundary();
            nut = distributeField<scalar>(nfd, gm.patches(), P.Lm, P.lp, P.procW, rank);
            nut.evaluateBoundary();
        }

        // SIMPLE loop
        // OF: unlisted field = not a criterion; converge only if a criterion was actually checked.
        int rcChecked = 0;
        auto ok = [&](scalar res, scalar ctl) { if (ctl < 0) return true; ++rcChecked; return res < ctl; };
        int iter = 0;
        bool converged = false;
        for (iter = 1; iter <= endTime && !converged; ++iter)
        {
            std::vector<scalar> nuEffCell(P.nCells(), nu);
            std::vector<std::vector<scalar>> nuEffBnd(P.lp.size());
            for (std::size_t pi = 0; pi < P.lp.size(); ++pi)
                nuEffBnd[pi].assign(P.lp[pi].size, nu);
            if (turbulent)
            {
                for (label c = 0; c < P.nCells(); ++c)
                    nuEffCell[c] = nu + nut.internal[c];
                for (std::size_t pi = 0; pi < P.lp.size(); ++pi)
                {
                    const auto& nv = nut.boundary[pi]->value();
                    for (label i = 0; i < P.lp[pi].size; ++i)
                        nuEffBnd[pi][i] = nu + nv[i];
                }
            }
            const ParStepResidual r = parallelSimpleStepLaminar(P, U, p, phi, nuEffCell, nuEffBnd, Ufd, relaxU, relaxP, tolU, tolP, boundedDiv);
            kepsilon::TurbResidual tres;
            if (turbulent) tres = kepsilon::parallelCorrect(P, U, k, eps, nut, phi, nu,
                                                          relaxEps, relaxK, std::fmin(tolK, tolE), boundedDiv, keCoeffs);
            if (master && (iter == 1 || iter % 50 == 0))
                std::printf("  iter %4d:  Ux %.3e  p %.3e%s\n", iter, r.Ux, r.p,
                            turbulent ? ("  k " + std::to_string(tres.k) + "  eps " + std::to_string(tres.eps)).c_str() : "");
            rcChecked = 0;
            converged = hasRC && ok(r.p, rcP) && ok(r.Ux, rcU) && (!turbulent || (ok(tres.k, rcK) && ok(tres.eps, rcE)));
            converged = converged && rcChecked > 0;   // OF's `checked` safety (simpleControl.C:57)
        }
        const int nIter = converged ? iter - 1 : endTime;
        if (master) std::printf(converged ? "SIMPLE solution converged in %d iterations\n"
                                           : "SIMPLE reached endTime (%d iterations)\n", nIter);

        // gather + write the converged time directory
        const std::vector<vector> Ug = gather(P, U.internal);
        const std::vector<scalar> pg = gather(P, p.internal);
        std::vector<scalar> kg, eg, ng;
        if (turbulent)
        {
            kg = gather(P, k.internal);
            eg = gather(P, eps.internal);
            ng = gather(P, nut.internal);
        }
        if (master)
        {
            const std::string outDir = caseDir + "/" + std::to_string(nIter);
            std::filesystem::create_directories(outDir);
            const std::string src = caseDir + "/" + startStr + "/";
            writeVolField(src + "U", outDir + "/U", Ug, gm.patches(), precision);
            writeVolField(src + "p", outDir + "/p", pg, gm.patches(), precision);
            if (turbulent)
            {
                writeVolField(src + "k", outDir + "/k", kg, gm.patches(), precision);
                writeVolField(src + "epsilon", outDir + "/epsilon", eg, gm.patches(), precision);
                writeVolField(src + "nut", outDir + "/nut", ng, gm.patches(), precision);
            }
            std::printf("written %s/{U,p%s}\n", outDir.c_str(), turbulent ? ",k,epsilon,nut" : "");
        }
    }
    catch (const std::exception& e)
    {
        if (master) std::fprintf(stderr, "brae_simpleFoam ERROR: %s\n", e.what());
        Pstream::finalize();
        return 1;
    }
    Pstream::finalize();
    return 0;
}
