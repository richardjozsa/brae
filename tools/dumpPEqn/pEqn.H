volScalarField rAU(1.0/UEqn.A());
surfaceScalarField rhorAUf("rhorAUf", fvc::interpolate(rho*rAU));
volVectorField HbyA(constrainHbyA(rAU*UEqn.H(), U, p));
tUEqn.clear();

bool closedVolume = false;
bool dumpedPCoeffsS = false;   // stage harness, subsonic branch

surfaceScalarField phiHbyA("phiHbyA", fvc::interpolate(rho)*fvc::flux(HbyA));

// dumpPEqn stage harness -- same quantities as pcEqn.H, so a case can be compared with `consistent no`.
// rAtU does not exist on this path (no SIMPLEC), so rAU stands in for both.
if (runTime.timeIndex() == braeDumpIter())
{
    volScalarField("stage_rAU",  rAU).write();
    volScalarField("stage_rAtU", rAU).write();
    volVectorField("stage_HbyA", HbyA).write();
    surfaceScalarField("stage_phi", phi).write();
    surfaceScalarField("stage_rhorAUf", rhorAUf).write();
    volScalarField("stage_rhorAU", rho*rAU).write();
    volScalarField("stage_rhoP", rho).write();
    volScalarField("stage_rAUp", rAU).write();
    surfaceScalarField("phiHbyA0", phiHbyA).write();
}
MRF.makeRelative(fvc::interpolate(rho), phiHbyA);

// Update the pressure BCs to ensure flux consistency
constrainPressure(p, rho, U, phiHbyA, rhorAUf, MRF);

if (simple.transonic())
{
    surfaceScalarField phid
    (
        "phid",
        (fvc::interpolate(psi)/fvc::interpolate(rho))*phiHbyA
    );

    phiHbyA -= fvc::interpolate(psi*p)*phiHbyA/fvc::interpolate(rho);

    while (simple.correctNonOrthogonal())
    {
        fvScalarMatrix pEqn
        (
            fvc::div(phiHbyA)
          + fvm::div(phid, p)
          - fvm::laplacian(rhorAUf, p)
          ==
            fvOptions(psi, p, rho.name())
        );

        // Relax the pressure equation to ensure diagonal-dominance
        pEqn.relax();

        pEqn.setReference
        (
            pressureControl.refCell(),
            pressureControl.refValue()
        );

        pEqn.solve();

        if (simple.finalNonOrthogonalIter())
        {
            phi = phiHbyA + pEqn.flux();
        }
    }
}
else
{
    closedVolume = adjustPhi(phiHbyA, U, p);

    while (simple.correctNonOrthogonal())
    {
        fvScalarMatrix pEqn
        (
            fvc::div(phiHbyA)
          - fvm::laplacian(rhorAUf, p)
          ==
            fvOptions(psi, p, rho.name())
        );

        // Stage harness: the pEqn coefficients and assembled system on the SUBSONIC branch too.
        if (runTime.timeIndex() == braeDumpIter())
        {
            dumpedPCoeffsS = true;
            volScalarField pIC
            (
                IOobject("stage_pIC", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
                mesh, dimensionedScalar(dimless, Zero), calculatedFvPatchScalarField::typeName
            );
            volScalarField pBC
            (
                IOobject("stage_pBC", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
                mesh, dimensionedScalar(dimless, Zero), calculatedFvPatchScalarField::typeName
            );
            forAll(mesh.boundary(), pi)
            {
                pIC.boundaryFieldRef()[pi] = pEqn.internalCoeffs()[pi];
                pBC.boundaryFieldRef()[pi] = pEqn.boundaryCoeffs()[pi];
            }
            pIC.write(); pBC.write();
            volScalarField pD
            (
                IOobject("stage_pD", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
                mesh, dimensionedScalar(dimless, Zero), zeroGradientFvPatchScalarField::typeName
            );
            volScalarField pSrc
            (
                IOobject("stage_pSrc", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
                mesh, dimensionedScalar(dimless, Zero), zeroGradientFvPatchScalarField::typeName
            );
            pD.primitiveFieldRef() = pEqn.D();
            scalarField rhs(pEqn.source());
            forAll(mesh.boundary(), pj)
            {
                const labelUList& fc = mesh.boundary()[pj].faceCells();
                const scalarField& bcp = pEqn.boundaryCoeffs()[pj];
                forAll(fc, i) rhs[fc[i]] += bcp[i];
            }
            pSrc.primitiveFieldRef() = rhs;
            pD.write(); pSrc.write();
            forAll(mesh.boundary(), pi)
            {
                const fvPatch& pp = mesh.boundary()[pi];
                if (!pp.size()) continue;
                scalar si = 0, sb = 0;
                forAll(pEqn.internalCoeffs()[pi], i)
                {
                    si += mag(pEqn.internalCoeffs()[pi][i]);
                    sb += mag(pEqn.boundaryCoeffs()[pi][i]);
                }
                Info<< "  pEqn patch " << pp.name() << " (" << pp.size() << " faces)"
                    << "  sum|iC| = " << si << "  sum|bC| = " << sb << endl;
            }
        }

        pEqn.setReference
        (
            pressureControl.refCell(),
            pressureControl.refValue()
        );

        pEqn.solve();

        if (simple.finalNonOrthogonalIter())
        {
            phi = phiHbyA + pEqn.flux();
        }
    }
}

#include "incompressible/continuityErrs.H"

if (runTime.timeIndex() == braeDumpIter())
{
    volScalarField("stage_pSolved", p).write();     // straight out of the linear solver
    surfaceScalarField("stage_phiSolved", phi).write();
}

// Explicitly relax pressure for momentum corrector
p.relax();

if (runTime.timeIndex() == braeDumpIter()) volScalarField("stage_pRelaxed", p).write();

U = HbyA - rAU*fvc::grad(p);
U.correctBoundaryConditions();
fvOptions.correct(U);

bool pLimited = pressureControl.limit(p);

// For closed-volume cases adjust the pressure and density levels
// to obey overall mass continuity
if (closedVolume)
{
    p += (initialMass - fvc::domainIntegrate(psi*p))
        /fvc::domainIntegrate(psi);
}

if (pLimited || closedVolume)
{
    p.correctBoundaryConditions();
}

rho = thermo.rho();

if (!simple.transonic())
{
    rho.relax();
}
