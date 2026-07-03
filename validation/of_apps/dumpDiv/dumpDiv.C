// Instrumentation: dump the convection matrix fvm::div(phi,T) (case's div scheme) and the
// lduMatrix matrix-vector product Amul of fvm::laplacian(1,T) on a known psi, for cf to
// validate fvm::div and SpMV against OpenFOAM.
#include "fvCFD.H"

static void dumpMatrix(const fvMesh& mesh, const fvScalarMatrix& M, const fileName& path)
{
    OFstream os(path);
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
        forAll(fp, i) os << M.internalCoeffs()[p][i] << ' ' << M.boundaryCoeffs()[p][i] << nl;
    }
}

int main(int argc, char *argv[])
{
    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"

    instantList times = runTime.times();
    runTime.setTime(times.last(), times.size() - 1);
    Info << "Reading T, phi at time " << runTime.timeName() << endl;

    volScalarField T
    (
        IOobject("T", runTime.timeName(), mesh, IOobject::MUST_READ, IOobject::NO_WRITE), mesh
    );
    surfaceScalarField phi
    (
        IOobject("phi", runTime.timeName(), mesh, IOobject::MUST_READ, IOobject::NO_WRITE), mesh
    );

    fvScalarMatrix Mdiv(fvm::div(phi, T));
    dumpMatrix(mesh, Mdiv, runTime.path()/"matrix_div.dat");

    // SpMV: raw lduMatrix Amul of the laplacian on a known psi (no coupled interfaces here).
    dimensionedScalar gamma("gamma", dimless, scalar(1));
    fvScalarMatrix Mlap(fvm::laplacian(gamma, T));
    solveScalarField psi(mesh.nCells());
    forAll(psi, c) psi[c] = 1.0 + 0.001*scalar(c);
    solveScalarField Apsi(mesh.nCells(), 0.0);
    const lduInterfaceFieldPtrsList interfaces(0);
    const FieldField<Field, scalar> bouCoeffs(0);
    Mlap.Amul(Apsi, psi, bouCoeffs, interfaces, 0);

    OFstream os(runTime.path()/"amul.dat");
    os.precision(16);
    os << mesh.nCells() << nl;
    forAll(psi, c) os << psi[c] << ' ' << Apsi[c] << nl;

    Info << "Wrote matrix_div.dat, amul.dat" << endl;
    return 0;
}
