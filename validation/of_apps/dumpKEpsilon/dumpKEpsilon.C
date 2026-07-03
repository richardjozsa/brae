// Instrumentation: dump grad(U), GbyNu = gradU && devTwoSymm(gradU), nut = Cmu*k^2/eps, and
// production G = nut*GbyNu, for cf to validate the k-epsilon production + eddy viscosity.
#include "fvCFD.H"

int main(int argc, char *argv[])
{
    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"

    instantList times = runTime.times();
    runTime.setTime(times.last(), times.size() - 1);
    Info << "time " << runTime.timeName() << endl;

    volVectorField U(IOobject("U", runTime.timeName(), mesh, IOobject::MUST_READ, IOobject::NO_WRITE), mesh);
    volScalarField k(IOobject("k", runTime.timeName(), mesh, IOobject::MUST_READ, IOobject::NO_WRITE), mesh);
    volScalarField epsilon(IOobject("epsilon", runTime.timeName(), mesh, IOobject::MUST_READ, IOobject::NO_WRITE), mesh);

    dimensionedScalar Cmu("Cmu", dimless, scalar(0.09));
    volTensorField gradU(fvc::grad(U));
    volScalarField GbyNu(gradU && devTwoSymm(gradU));
    volScalarField nut(Cmu * sqr(k) / epsilon);
    volScalarField G(nut * GbyNu);

    OFstream os(runTime.path()/"kepsilon.dat");
    os.precision(16);
    os << mesh.nCells() << nl;
    forAll(gradU.primitiveField(), c) {
        const tensor& t = gradU[c];
        os << t.xx() << ' ' << t.xy() << ' ' << t.xz() << ' '
           << t.yx() << ' ' << t.yy() << ' ' << t.yz() << ' '
           << t.zx() << ' ' << t.zy() << ' ' << t.zz() << nl;
    }
    forAll(GbyNu.primitiveField(), c) os << GbyNu[c] << nl;
    forAll(nut.primitiveField(),   c) os << nut[c]   << nl;
    forAll(G.primitiveField(),     c) os << G[c]     << nl;
    Info << "kepsilon dumped" << endl;
    return 0;
}
