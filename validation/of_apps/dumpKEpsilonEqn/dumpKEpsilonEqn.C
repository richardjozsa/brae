// Instrumentation: assemble the k-epsilon transport matrices (BEFORE relax/wall-constraint/solve)
// and dump them for cf to validate fvm::Sp + the equation assembly.
//   epsEqn = div(phi,eps) - laplacian(DepsEff,eps) + Sp(C2*eps/k,eps) == C1*Cmu*k*GbyNu
//   kEqn   = div(phi,k)   - laplacian(DkEff,k)     + Sp(eps/k,k)       == G
#include "fvCFD.H"

static void dumpMatrix(const fvMesh& mesh, const fvScalarMatrix& M, const fileName& path)
{
    OFstream os(path);
    os.precision(16);
    os << mesh.nCells() << ' ' << mesh.nInternalFaces() << ' ' << mesh.boundary().size() << nl;
    forAll(M.diag(),   c) os << M.diag()[c]   << nl;
    forAll(M.upper(),  f) os << M.upper()[f]  << nl;
    const scalarField& low = M.lower();
    forAll(low,        f) os << low[f]         << nl;
    forAll(M.source(), c) os << M.source()[c] << nl;
    forAll(mesh.boundary(), p) {
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

    volVectorField U(IOobject("U", runTime.timeName(), mesh, IOobject::MUST_READ, IOobject::NO_WRITE), mesh);
    volScalarField k(IOobject("k", runTime.timeName(), mesh, IOobject::MUST_READ, IOobject::NO_WRITE), mesh);
    volScalarField epsilon(IOobject("epsilon", runTime.timeName(), mesh, IOobject::MUST_READ, IOobject::NO_WRITE), mesh);
    volScalarField nut(IOobject("nut", runTime.timeName(), mesh, IOobject::MUST_READ, IOobject::NO_WRITE), mesh);
    surfaceScalarField phi(IOobject("phi", runTime.timeName(), mesh, IOobject::MUST_READ, IOobject::NO_WRITE), mesh);

    dimensionedScalar nu("nu", dimViscosity, scalar(1e-5));
    dimensionedScalar Cmu("Cmu", dimless, scalar(0.09));
    dimensionedScalar C1("C1", dimless, scalar(1.44));
    dimensionedScalar C2("C2", dimless, scalar(1.92));
    dimensionedScalar sigmak("sigmak", dimless, scalar(1.0));
    dimensionedScalar sigmaEps("sigmaEps", dimless, scalar(1.3));

    volTensorField gradU(fvc::grad(U));
    volScalarField GbyNu(gradU && devTwoSymm(gradU));
    volScalarField G(nut * GbyNu);
    volScalarField DkEff(nut / sigmak + nu);
    volScalarField DepsEff(nut / sigmaEps + nu);

    fvScalarMatrix epsEqn
    (
        fvm::div(phi, epsilon) - fvm::laplacian(DepsEff, epsilon)
      + fvm::Sp(C2 * epsilon / k, epsilon)
     == C1 * Cmu * k * GbyNu
    );
    fvScalarMatrix kEqn
    (
        fvm::div(phi, k) - fvm::laplacian(DkEff, k)
      + fvm::Sp(epsilon / k, k)
     == G
    );

    dumpMatrix(mesh, epsEqn, runTime.path()/"epseqn.dat");
    dumpMatrix(mesh, kEqn,   runTime.path()/"keqn.dat");
    Info << "k/epsilon matrices dumped" << endl;
    return 0;
}
