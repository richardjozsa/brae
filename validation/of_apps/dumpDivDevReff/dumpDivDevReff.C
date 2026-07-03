// Instrumentation: dump the EXPLICIT part of turbulence->divDevReff(U), i.e.
//   fvc::div(nuEff*dev2(T(fvc::grad(U)))),
// plus nuEff (internal + boundary) and grad(U) (internal), so cf can validate its new
// gradUBoundary + fvc::div(volTensorField) operators against OpenFOAM term-by-term.
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

    volVectorField U(IOobject("U", runTime.timeName(), mesh, IOobject::MUST_READ, IOobject::NO_WRITE), mesh);
    surfaceScalarField phi(IOobject("phi", runTime.timeName(), mesh, IOobject::MUST_READ, IOobject::NO_WRITE), mesh);

    singlePhaseTransportModel laminarTransport(U, phi);
    autoPtr<incompressible::turbulenceModel> turbulence
    (
        incompressible::turbulenceModel::New(U, phi, laminarTransport)
    );

    const volScalarField nuEff(turbulence->nuEff());
    const volTensorField gradU(fvc::grad(U));
    const volTensorField sigma((nuEff)*dev2(T(gradU)));
    const volVectorField divPart(fvc::div(sigma));

    OFstream os(runTime.path()/"divdevreff.dat");
    os.precision(16);
    os << mesh.nCells() << nl;
    forAll(divPart, c) os << divPart[c].x() << ' ' << divPart[c].y() << ' ' << divPart[c].z() << nl;
    forAll(nuEff,   c) os << nuEff[c] << nl;
    forAll(gradU,   c)
    {
        const tensor& t = gradU[c];
        os << t.xx() << ' ' << t.xy() << ' ' << t.xz() << ' '
           << t.yx() << ' ' << t.yy() << ' ' << t.yz() << ' '
           << t.zx() << ' ' << t.zy() << ' ' << t.zz() << nl;
    }
    // Per-patch boundary gradU and sigma face fields (what fvc::div consumes at the boundary).
    OFstream ob(runTime.path()/"divdevreff_bnd.dat");
    ob.precision(16);
    forAll(mesh.boundary(), patchi)
    {
        const fvPatch& p = mesh.boundary()[patchi];
        const tensorField& gb = gradU.boundaryField()[patchi];
        const tensorField& sb = sigma.boundaryField()[patchi];
        ob << p.name() << ' ' << p.size() << nl;
        forAll(p, i)
        {
            const label c = p.faceCells()[i];
            ob << c << ' '
               << gb[i].xx() << ' ' << gb[i].xy() << ' ' << gb[i].yx() << ' ' << gb[i].yy() << ' '
               << sb[i].xx() << ' ' << sb[i].xy() << ' ' << sb[i].yx() << ' ' << sb[i].yy() << nl;
        }
    }
    Info << "divDevReff explicit part + boundary dumped" << endl;
    return 0;
}
