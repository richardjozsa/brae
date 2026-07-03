// Instrumentation: solve -fvm::laplacian(1,T)=0 (a PD diffusion problem set by T's BCs) with
// the case's linear solver (PCG/DIC), and dump the solution + solver performance, for cf to
// validate its PCG against OpenFOAM (same matrix, same iteration).
#include "fvCFD.H"

int main(int argc, char *argv[])
{
    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"

    instantList times = runTime.times();
    runTime.setTime(times.last(), times.size() - 1);
    Info << "Solving at time " << runTime.timeName() << endl;

    volScalarField T
    (
        IOobject("T", runTime.timeName(), mesh, IOobject::MUST_READ, IOobject::NO_WRITE), mesh
    );

    dimensionedScalar gamma("gamma", dimless, scalar(1));
    fvScalarMatrix M(-fvm::laplacian(gamma, T));
    SolverPerformance<scalar> perf = M.solve();

    OFstream os(runTime.path()/"solution.dat");
    os.precision(16);
    os << mesh.nCells() << nl;
    os << perf.initialResidual() << ' ' << perf.finalResidual() << ' ' << perf.nIterations() << nl;
    forAll(T.primitiveField(), c) os << T[c] << nl;

    Info << "init=" << perf.initialResidual() << " final=" << perf.finalResidual()
         << " iters=" << perf.nIterations() << endl;
    return 0;
}
