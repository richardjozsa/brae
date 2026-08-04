// Demo instrumentation for the AMG-PCG showcase (demo/amgpcg).
//
// Dumps, as CSV, everything the replay animation needs from the REAL OpenFOAM side:
//   mesh.csv          cell centres + the structured (i,j) index, so the animation can draw the mesh
//   matrix.csv        the real lduMatrix of -fvm::laplacian(1,T): diag/upper/lower + addressing
//   rhs.csv           the real source vector (boundary contributions folded in)
//   of_hierarchy.csv  the REAL faceAreaPair agglomeration, level by level, cell by cell
//   of_cycles.csv     the REAL GAMG state after each V-cycle (psi and residual, per cell)
//
// The per-cycle trace is produced by re-solving from psi=0 with maxIter = 1, 2, 3, ...
// GAMG as a standalone solver is a deterministic stationary iteration, so the state after
// the k-th cycle of a long solve is exactly the state a maxIter=k solve stops at. That
// makes every number in of_cycles.csv a genuine OpenFOAM output, not a reimplementation.

#include "fvCFD.H"
#include "GAMGAgglomeration.H"

// ------------------------------------------------------------------------------------- //

static void writeMesh
(
    const fvMesh& mesh,
    const fileName& path
)
{
    OFstream os(path);
    os.precision(16);
    os << "cell,x,y,z,volume" << nl;

    forAll(mesh.C(), c)
    {
        os  << c << ','
            << mesh.C()[c].x() << ','
            << mesh.C()[c].y() << ','
            << mesh.C()[c].z() << ','
            << mesh.V()[c] << nl;
    }

    Info << "wrote " << path << endl;
}

// ------------------------------------------------------------------------------------- //

static void writeMatrix
(
    const fvMesh& mesh,
    const fvScalarMatrix& M,
    const fileName& path
)
{
    const labelUList& lowerAddr = mesh.lduAddr().lowerAddr();
    const labelUList& upperAddr = mesh.lduAddr().upperAddr();

    // The solved system is the lduMatrix WITH the boundary contributions folded in, exactly
    // what lduMatrix::solve() does internally (addBoundaryDiag / addBoundarySource). Dumping
    // the bare diag would give the pure-Neumann singular operator, not the system anyone solves.
    scalarField D(M.diag());

    forAll(mesh.boundary(), p)
    {
        const labelUList& faceCells = mesh.boundary()[p].faceCells();

        forAll(faceCells, i)
        {
            D[faceCells[i]] += M.internalCoeffs()[p][i];
        }
    }

    OFstream os(path);
    os.precision(16);
    os << "kind,index,owner,neighbour,value" << nl;

    forAll(D, c)
    {
        os << "diag," << c << ',' << c << ',' << c << ',' << D[c] << nl;
    }

    const scalarField& lower = M.lower();
    const scalarField& upper = M.upper();

    forAll(upper, f)
    {
        os  << "upper," << f << ',' << lowerAddr[f] << ',' << upperAddr[f] << ','
            << upper[f] << nl;
    }

    forAll(lower, f)
    {
        os  << "lower," << f << ',' << lowerAddr[f] << ',' << upperAddr[f] << ','
            << lower[f] << nl;
    }

    Info << "wrote " << path << endl;
}

// ------------------------------------------------------------------------------------- //

static void writeRhs
(
    const fvMesh& mesh,
    const fvScalarMatrix& M,
    const fileName& path
)
{
    scalarField S(M.source());

    forAll(mesh.boundary(), p)
    {
        const labelUList& faceCells = mesh.boundary()[p].faceCells();

        forAll(faceCells, i)
        {
            S[faceCells[i]] += M.boundaryCoeffs()[p][i];
        }
    }

    OFstream os(path);
    os.precision(16);
    os << "cell,b" << nl;

    forAll(S, c)
    {
        os << c << ',' << S[c] << nl;
    }

    Info << "wrote " << path << endl;
}

// ------------------------------------------------------------------------------------- //
// The real faceAreaPair hierarchy. restrictAddressing(L)[c] is the coarse cell that fine
// cell c of level L is agglomerated into. We also chain the maps back to the finest grid,
// so the animation can colour the original 20 cells by their aggregate at any level.

static void writeHierarchy
(
    const fvMesh& mesh,
    const dictionary& solverDict,
    const fileName& path
)
{
    const GAMGAgglomeration& agglom = GAMGAgglomeration::New(mesh, solverDict);

    OFstream os(path);
    os << "level,fine_cell,coarse_cell,finest_cell" << nl;

    // finestOf[c] on level L = the list of level-0 cells that ended up in level-L cell c.
    // Carried forward as a per-level map: chain[L][fineCellOfLevel0] = cell index on level L.
    labelList chain(mesh.nCells());

    forAll(chain, c)
    {
        chain[c] = c;
    }

    for (label lev = 0; lev < agglom.size(); ++lev)
    {
        const labelField& restrict = agglom.restrictAddressing(lev);

        forAll(restrict, c)
        {
            // one row per (level, fine cell) pair, plus one row per finest cell for painting
            os << lev << ',' << c << ',' << restrict[c] << ',' << -1 << nl;
        }

        forAll(chain, c0)
        {
            chain[c0] = restrict[chain[c0]];
            os << lev << ',' << -1 << ',' << chain[c0] << ',' << c0 << nl;
        }

        Info<< "level " << lev << ": " << restrict.size() << " -> "
            << (max(restrict) + 1) << " cells" << endl;
    }

    Info << "wrote " << path << " (" << agglom.size() << " coarse levels)" << endl;
}

// ------------------------------------------------------------------------------------- //
// The real GAMG per-cycle trace: k = 0 .. maxCycles, each a fresh solve from psi = 0 with
// maxIter = k, dumping psi and the true residual r = b - A psi for every cell.

static void writeCycles
(
    const fvMesh& mesh,
    volScalarField& T,
    const dimensionedScalar& gamma,
    const dictionary& solverDict,
    const label maxCycles,
    const fileName& path
)
{
    OFstream os(path);
    os.precision(16);
    os << "cycle,cell,psi,residual,resnorm" << nl;

    const scalarField T0(T.primitiveField().size(), scalar(0));

    for (label k = 0; k <= maxCycles; ++k)
    {
        T.primitiveFieldRef() = T0;
        T.correctBoundaryConditions();

        fvScalarMatrix M(-fvm::laplacian(gamma, T));

        scalar resNorm = 0;

        if (k > 0)
        {
            dictionary d(solverDict);
            d.set("maxIter", k);
            d.set("tolerance", scalar(0));
            d.set("relTol", scalar(0));

            SolverPerformance<scalar> perf = M.solve(d);
            resNorm = perf.finalResidual();
        }
        else
        {
            resNorm = 1;
        }

        const scalarField res(M.residual());

        forAll(T.primitiveField(), c)
        {
            os  << k << ',' << c << ','
                << T.primitiveField()[c] << ','
                << res[c] << ','
                << resNorm << nl;
        }
    }

    Info << "wrote " << path << " (" << maxCycles << " cycles)" << endl;
}

// ------------------------------------------------------------------------------------- //

int main(int argc, char *argv[])
{
    argList::addOption("maxCycles", "int", "number of GAMG cycles to trace (default 30)");
    argList::addBoolOption
    (
        "measure",
        "solve once to 1e-8 and report the cycle count; write no CSV. "
        "The per-cycle dump is O(cells x cycles) rows, unusable past ~1e5 cells."
    );

    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"

    const label maxCycles = args.getOrDefault<label>("maxCycles", 30);

    volScalarField T
    (
        IOobject("T", runTime.timeName(), mesh, IOobject::MUST_READ, IOobject::NO_WRITE),
        mesh
    );

    const dimensionedScalar gamma("gamma", dimless, scalar(1));

    // -laplacian so the operator is positive-definite, the same sign convention the
    // pressure equation uses and the one brae's AMG assumes.
    fvScalarMatrix M(-fvm::laplacian(gamma, T));

    if (args.found("measure"))
    {
        const dictionary& measureDict = mesh.solverDict("T");
        const GAMGAgglomeration& agglom = GAMGAgglomeration::New(mesh, measureDict);

        Info << "  levels: " << agglom.size() << endl;

        for (label lev = 0; lev < agglom.size(); ++lev)
        {
            const labelField& restrict = agglom.restrictAddressing(lev);
            Info << "    " << restrict.size() << " -> " << (max(restrict) + 1) << endl;
        }

        dictionary d(measureDict);
        d.set("tolerance", scalar(1e-8));
        d.set("relTol", scalar(0));
        d.set("maxIter", label(5000));

        // warm the caches (agglomeration is cached, so the first solve pays for it)
        {
            volScalarField warm(T);
            fvScalarMatrix W(-fvm::laplacian(gamma, warm));
            W.solve(d);
        }

        scalar bestMs = GREAT;
        SolverPerformance<scalar> perf;

        for (label rep = 0; rep < 5; ++rep)
        {
            T.primitiveFieldRef() = scalar(0);
            T.correctBoundaryConditions();
            fvScalarMatrix R(-fvm::laplacian(gamma, T));

            const scalar t0 = mesh.time().elapsedCpuTime();
            perf = R.solve(d);
            const scalar dt = (mesh.time().elapsedCpuTime() - t0) * 1000.0;

            bestMs = min(bestMs, dt);
        }

        Info<< "BENCH of_gamg  solve=" << bestMs << "ms"
            << "  ms/cycle=" << bestMs / max(label(1), perf.nIterations()) << endl;

        Info<< "MEASURE of_gamg  cells=" << mesh.nCells()
            << "  iters=" << perf.nIterations()
            << "  init=" << perf.initialResidual()
            << "  final=" << perf.finalResidual() << endl;

        return 0;
    }

    const fileName outDir(runTime.path()/"trace");
    mkDir(outDir);

    writeMesh(mesh, outDir/"mesh.csv");
    writeMatrix(mesh, M, outDir/"matrix.csv");
    writeRhs(mesh, M, outDir/"rhs.csv");

    const dictionary& solverDict = mesh.solverDict("T");
    writeHierarchy(mesh, solverDict, outDir/"of_hierarchy.csv");
    writeCycles(mesh, T, gamma, solverDict, maxCycles, outDir/"of_cycles.csv");

    Info << "done" << endl;
    return 0;
}

// ************************************************************************* //
