/*---------------------------------------------------------------------------*\
    dumpScalarMatrix

    Dumps the assembled fvScalarMatrix (diagonal and source) for a scalar
    transport equation, so another code's assembly can be compared against
    OpenFOAM's cell by cell.

    Deliberately NOT the full kOmegaSST omega equation: F1/beta/gamma/CDkOmega
    are protected members of kOmegaSSTBase, so reconstructing them here would
    duplicate model internals and test my copy rather than OpenFOAM's. The
    machinery actually under investigation is generic -- the bounded prefix, the
    upwind matrix, the linearUpwind deferred correction, and the laplacian -- so
    this builds exactly that, on the real converged field, with a uniform
    diffusivity:

        fvm::div(phi, psi) - fvm::laplacian(gamma, psi)

    div(phi,<field>) and the laplacian scheme come from the case's own
    fvSchemes, so `bounded Gauss linearUpwind grad(omega)` is honoured.

    phi is rebuilt as fvc::flux(U) rather than read, so the comparison uses a
    flux both codes can reproduce identically from the same U.

    Usage:  dumpScalarMatrix -case <dir> [-field omega] [-gamma 1e-3]
    Writes: Ddiag, Dsource (volScalarFields) in the latest time directory.
\*---------------------------------------------------------------------------*/

#include "fvCFD.H"

int main(int argc, char *argv[])
{
    argList::addOption("field", "name", "scalar field to assemble for (default omega)");
    argList::addOption("gamma", "value", "uniform diffusivity (default 1e-3)");

    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"

    const word fieldName = args.getOrDefault<word>("field", "omega");
    const scalar gammaVal = args.getOrDefault<scalar>("gamma", 1e-3);

    instantList times = runTime.times();
    runTime.setTime(times.last(), times.size() - 1);
    Info<< "Time = " << runTime.timeName() << nl
        << "field = " << fieldName << ", uniform gamma = " << gammaVal << endl;
    mesh.readUpdate();

    volVectorField U
    (
        IOobject("U", runTime.timeName(), mesh, IOobject::MUST_READ, IOobject::NO_WRITE),
        mesh
    );
    volScalarField psi
    (
        IOobject(fieldName, runTime.timeName(), mesh, IOobject::MUST_READ, IOobject::NO_WRITE),
        mesh
    );

    surfaceScalarField phi("phi", fvc::flux(U));
    dimensionedScalar gamma("gamma", dimViscosity, gammaVal);

    fvScalarMatrix eqn
    (
        fvm::div(phi, psi)
      - fvm::laplacian(gamma, psi)
    );

    // Raw, pre-boundary-fold diagonal and source -- the same stage the other code dumps.
    volScalarField Ddiag
    (
        IOobject("Ddiag", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::AUTO_WRITE),
        mesh,
        dimensionedScalar(dimless, Zero),
        zeroGradientFvPatchScalarField::typeName
    );
    volScalarField Dsource
    (
        IOobject("Dsource", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::AUTO_WRITE),
        mesh,
        dimensionedScalar(dimless, Zero),
        zeroGradientFvPatchScalarField::typeName
    );

    // The boundary coefficients: how a BC actually enters the matrix. internalCoeffs goes to the
    // diagonal of the adjacent cell, boundaryCoeffs to its source. These are what a "the BC is present
    // but not applied where the discretisation needs it" bug corrupts, and they are NOT exercised by a
    // diagonal comparison on plain BCs.
    volScalarField ICoeff
    (
        IOobject("ICoeff", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::AUTO_WRITE),
        mesh,
        dimensionedScalar(dimless, Zero),
        calculatedFvPatchScalarField::typeName
    );
    volScalarField BCoeff
    (
        IOobject("BCoeff", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::AUTO_WRITE),
        mesh,
        dimensionedScalar(dimless, Zero),
        calculatedFvPatchScalarField::typeName
    );
    forAll(mesh.boundary(), pi)
    {
        ICoeff.boundaryFieldRef()[pi] = eqn.internalCoeffs()[pi];
        BCoeff.boundaryFieldRef()[pi] = eqn.boundaryCoeffs()[pi];
    }
    ICoeff.write();
    BCoeff.write();
    forAll(mesh.boundary(), pi)
    {
        const fvPatch& pp = mesh.boundary()[pi];
        if (!pp.size()) continue;
        scalar si = 0, sb = 0;
        forAll(eqn.internalCoeffs()[pi], i)
        {
            si += mag(eqn.internalCoeffs()[pi][i]);
            sb += mag(eqn.boundaryCoeffs()[pi][i]);
        }
        Info<< "  patch " << pp.name() << " (" << psi.boundaryField()[pi].type() << ", "
            << pp.size() << " faces): sum|internalCoeffs| = " << si
            << ", sum|boundaryCoeffs| = " << sb << endl;
    }

    Ddiag.primitiveFieldRef()   = eqn.diag();
    Dsource.primitiveFieldRef() = eqn.source();
    Ddiag.correctBoundaryConditions();
    Dsource.correctBoundaryConditions();

    Ddiag.write();
    Dsource.write();

    scalar dmin = GREAT, dmax = -GREAT, sd = 0, ss = 0;
    forAll(eqn.diag(), c)
    {
        dmin = min(dmin, eqn.diag()[c]);
        dmax = max(dmax, eqn.diag()[c]);
        sd  += sqr(eqn.diag()[c] * psi[c]);
        ss  += sqr(eqn.source()[c]);
    }
    Info<< "diag[" << dmin << ", " << dmax << "]" << nl
        << "|diag*psi| = " << Foam::sqrt(sd) << nl
        << "|source|   = " << Foam::sqrt(ss) << nl
        << "wrote Ddiag and Dsource" << endl;

    Info<< "End" << endl;
    return 0;
}
