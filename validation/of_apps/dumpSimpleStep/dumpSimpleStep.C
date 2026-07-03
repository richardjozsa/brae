// Instrumentation: one SIMPLE iteration with the full laminar momentum (incompressible
// divDevReff): div(phi,U) - laplacian(nu,U) - div(nu*dev2(T(grad U))) (Gauss upwind), matching
// cf::simpleStep, from the latest time. Dumps the resulting p, U, phi for cf to validate.
#include "fvCFD.H"
#include "constrainHbyA.H"

int main(int argc, char *argv[])
{
    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"

    instantList times = runTime.times();
    runTime.setTime(times.last(), times.size() - 1);
    Info << "step at time " << runTime.timeName() << endl;

    volScalarField p(IOobject("p", runTime.timeName(), mesh, IOobject::MUST_READ, IOobject::NO_WRITE), mesh);
    p.storePrevIter();   // pressure relaxation relaxes toward the iteration-start value
    volVectorField U(IOobject("U", runTime.timeName(), mesh, IOobject::MUST_READ, IOobject::NO_WRITE), mesh);
    surfaceScalarField phi(IOobject("phi", runTime.timeName(), mesh, IOobject::MUST_READ, IOobject::AUTO_WRITE), mesh);
    dimensionedScalar nu("nu", dimViscosity, scalar(1e-5));

    // --- momentum (full incompressible stress, = laminar turbulence->divDevReff(U))
    fvVectorMatrix UEqn
    (
        fvm::div(phi, U)
      - fvm::laplacian(nu, U)
      - fvc::div(nu*dev2(T(fvc::grad(U))))
    );
    UEqn.relax();
    solve(UEqn == -fvc::grad(p));

    // --- pressure
    volScalarField rAU(1.0/UEqn.A());
    volVectorField HbyA(constrainHbyA(rAU*UEqn.H(), U, p));
    surfaceScalarField phiHbyA(fvc::flux(HbyA));
    fvScalarMatrix pEqn(fvm::laplacian(rAU, p) == fvc::div(phiHbyA));
    pEqn.solve();
    phi = phiHbyA - pEqn.flux();
    p.relax(0.3);   // explicit factor (standalone app has no SIMPLE-loop final-iteration state)
    U = HbyA - rAU*fvc::grad(p);
    U.correctBoundaryConditions();

    OFstream os(runTime.path()/"step.dat");
    os.precision(16);
    os << mesh.nCells() << ' ' << mesh.nInternalFaces() << ' ' << mesh.boundary().size() << nl;
    forAll(p.primitiveField(), c) os << p[c] << nl;
    forAll(U.primitiveField(), c) { const vector& u = U[c]; os << u.x() << ' ' << u.y() << ' ' << u.z() << nl; }
    forAll(phi.primitiveField(), f) os << phi[f] << nl;
    forAll(mesh.boundary(), pp) {
        const fvsPatchScalarField& pf = phi.boundaryField()[pp];
        os << mesh.boundary()[pp].name() << ' ' << pf.size() << nl;
        forAll(pf, i) os << pf[i] << nl;
    }
    Info << "step dumped" << endl;
    return 0;
}
