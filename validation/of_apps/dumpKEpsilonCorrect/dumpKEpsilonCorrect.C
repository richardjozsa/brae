// Instrumentation: create the incompressible kEpsilon model and call correct() once, then dump
// the resulting k, epsilon, nut, for cf to validate its full turbulence correct() (wall
// functions, constraint, solve, bound) against the real OpenFOAM model.
#include "fvCFD.H"
#include "singlePhaseTransportModel.H"
#include "turbulentTransportModel.H"

int main(int argc, char *argv[])
{
    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"

    instantList times = runTime.times();
    runTime.setTime(times.last(), times.size() - 1);
    Info << "correct() at time " << runTime.timeName() << endl;

    volScalarField p(IOobject("p", runTime.timeName(), mesh, IOobject::MUST_READ, IOobject::NO_WRITE), mesh);
    volVectorField U(IOobject("U", runTime.timeName(), mesh, IOobject::MUST_READ, IOobject::NO_WRITE), mesh);
    surfaceScalarField phi(IOobject("phi", runTime.timeName(), mesh, IOobject::MUST_READ, IOobject::NO_WRITE), mesh);

    singlePhaseTransportModel laminarTransport(U, phi);
    autoPtr<incompressible::turbulenceModel> turbulence
    (
        incompressible::turbulenceModel::New(U, phi, laminarTransport)
    );

    turbulence->correct();

    const volScalarField& k       = mesh.lookupObject<volScalarField>("k");
    const volScalarField& epsilon = mesh.lookupObject<volScalarField>("epsilon");
    const volScalarField& nut     = mesh.lookupObject<volScalarField>("nut");

    OFstream os(runTime.path()/"correct.dat");
    os.precision(16);
    os << mesh.nCells() << nl;
    forAll(k.primitiveField(),       c) os << k[c]       << nl;
    forAll(epsilon.primitiveField(), c) os << epsilon[c] << nl;
    forAll(nut.primitiveField(),     c) os << nut[c]     << nl;
    Info << "correct dumped" << endl;
    return 0;
}
