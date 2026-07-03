// Instrumentation utility: build fvm::laplacian(gamma=1, T) on a case and dump the raw
// lduMatrix (diag/upper/lower/source) + per-patch internalCoeffs/boundaryCoeffs, for cf to
// validate its matrix assembly cell-by-cell. Uses the case's fvSchemes laplacian scheme.
#include "fvCFD.H"

int main(int argc, char *argv[])
{
    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"

    instantList times = runTime.times();
    runTime.setTime(times.last(), times.size() - 1);
    Info << "Reading T at time " << runTime.timeName() << endl;

    volScalarField T
    (
        IOobject("T", runTime.timeName(), mesh, IOobject::MUST_READ, IOobject::NO_WRITE),
        mesh
    );

    dimensionedScalar gamma("gamma", dimless, scalar(1));
    fvScalarMatrix M(fvm::laplacian(gamma, T));

    OFstream os(runTime.path()/"matrix.dat");
    os.precision(16);
    os << mesh.nCells() << ' ' << mesh.nInternalFaces() << ' ' << mesh.boundary().size() << nl;

    forAll(M.diag(),   c) os << M.diag()[c]   << nl;
    forAll(M.upper(),  f) os << M.upper()[f]  << nl;
    const scalarField& lower = M.lower();
    forAll(lower,      f) os << lower[f]       << nl;
    forAll(M.source(), c) os << M.source()[c] << nl;

    forAll(mesh.boundary(), p)
    {
        const fvPatch& fp = mesh.boundary()[p];
        os << fp.name() << ' ' << fp.size() << nl;
        forAll(fp, i)
            os << M.internalCoeffs()[p][i] << ' ' << M.boundaryCoeffs()[p][i] << nl;
    }

    Info << "Wrote " << (runTime.path()/"matrix.dat") << endl;
    return 0;
}
