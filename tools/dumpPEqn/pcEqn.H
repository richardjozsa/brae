rho = thermo.rho();

volScalarField rAU(1.0/UEqn.A());
volScalarField rAtU(1.0/(1.0/rAU - UEqn.H1()));
volVectorField HbyA(constrainHbyA(rAU*UEqn.H(), U, p));
tUEqn.clear();

bool closedVolume = false;
bool dumpedPCoeffs = false;   // stage harness: dump the pEqn coefficients once, on the first non-orth pass

surfaceScalarField phiHbyA("phiHbyA", fvc::interpolate(rho)*fvc::flux(HbyA));
MRF.makeRelative(fvc::interpolate(rho), phiHbyA);

volScalarField rhorAtU("rhorAtU", rho*rAtU);

// dumpPEqn stage harness: the MOMENTUM stage's outputs, before any pressure work touches them. These are
// upstream of phid/phiHbyA and are what a stage-by-stage comparison needs first (see
// cudafoam/rhosimplefoam-restage-plan.md). Written at iteration 1 only.
if (runTime.timeIndex() == braeDumpIter())
{
    volScalarField("stage_rAU",  rAU).write();
    volScalarField("stage_rAtU", rAtU).write();
    volVectorField("stage_HbyA", HbyA).write();
    surfaceScalarField("stage_phi", phi).write();
}

// Update the pressure BCs to ensure flux consistency
constrainPressure(p, rho, U, phiHbyA, rhorAtU, MRF);

if (simple.transonic())
{
    if (runTime.timeIndex() == braeDumpIter()) surfaceScalarField("phiHbyA0", phiHbyA).write();

    surfaceScalarField phid
    (
        "phid",
        (fvc::interpolate(psi)/fvc::interpolate(rho))*phiHbyA
    );

    phiHbyA +=
        fvc::interpolate(rho*(rAtU - rAU))*fvc::snGrad(p)*mesh.magSf()
      - fvc::interpolate(psi*p)*phiHbyA/fvc::interpolate(rho);

    HbyA -= (rAU - rAtU)*fvc::grad(p);

    if (runTime.timeIndex() == braeDumpIter())
    {
        // dumpPEqn: the transonic pressure equation's INPUTS, so another code can be compared on them
        // BEFORE the matrix is built. If these match and the answer still differs, the fault is in the
        // matrix or the flux reconstruction, not upstream.
        phid.write();
        phiHbyA.write();
        rhorAtU.write();
        volScalarField("rAtU_", rAtU).write();
        volScalarField("psi_", psi).write();
        volScalarField("rho_", rho).write();
    }

    while (simple.correctNonOrthogonal())
    {
        fvScalarMatrix pEqn
        (
            fvc::div(phiHbyA)
          + fvm::div(phid, p)
          - fvm::laplacian(rhorAtU, p)
         ==
            fvOptions(psi, p, rho.name())
        );

        // Relax the pressure equation to maintain diagonal dominance
        pEqn.relax();

        // Stage harness: the PRESSURE equation's boundary coefficients, per patch, on the FIRST
        // non-orthogonal pass. relax() does not touch internalCoeffs/boundaryCoeffs (it moves the diagonal
        // and the source), so this is comparable against brae's pIC/pBC either side of its own relax.
        if (runTime.timeIndex() == braeDumpIter() && !dumpedPCoeffs)
        {
            dumpedPCoeffs = true;
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
            pIC.write();
            pBC.write();

            // The assembled LINEAR SYSTEM: D() is the diagonal INCLUDING internalCoeffs, and the full RHS
            // is source() plus the boundary contribution. These are what brae's diagCp_/bp_ hold after
            // deviceFoldPressure, so they are the like-for-like comparison (up to brae's overall sign,
            // whose matrix is the negative of this one).
            {
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
                // addBoundarySource() is protected, so the boundary contribution is applied explicitly:
                // boundaryCoeffs land on the adjacent cell of each patch face. Same arithmetic, public API.
                scalarField rhs(pEqn.source());
                forAll(mesh.boundary(), pj)
                {
                    const labelUList& fc = mesh.boundary()[pj].faceCells();
                    const scalarField& bcp = pEqn.boundaryCoeffs()[pj];
                    forAll(fc, i) rhs[fc[i]] += bcp[i];
                }
                pSrc.primitiveFieldRef() = rhs;
                pD.write();
                pSrc.write();
                Info<< "  pEqn D():   sum " << gSum(pEqn.D())
                    << "  sum|.| " << gSum(mag(pEqn.D())) << endl;
                Info<< "  pEqn RHS:   sum " << gSum(rhs)
                    << "  sum|.| " << gSum(mag(rhs)) << endl;
            }
            forAll(mesh.boundary(), pi)
            {
                const fvPatch& pp = mesh.boundary()[pi];
                if (!pp.size()) continue;
                scalar si = 0, sb = 0, ri = 0, rb = 0;
                forAll(pEqn.internalCoeffs()[pi], i)
                {
                    si += mag(pEqn.internalCoeffs()[pi][i]);
                    sb += mag(pEqn.boundaryCoeffs()[pi][i]);
                    ri += pEqn.internalCoeffs()[pi][i];
                    rb += pEqn.boundaryCoeffs()[pi][i];
                }
                Info<< "  pEqn patch " << pp.name() << " (" << pp.size() << " faces)"
                    << "  sum|iC| = " << si << " (signed " << ri << ")"
                    << "  sum|bC| = " << sb << " (signed " << rb << ")" << endl;
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
            if (runTime.timeIndex() == braeDumpIter())
            {
                surfaceScalarField pf("pEqnFlux", pEqn.flux());
                pf.write();
            }
            phi = phiHbyA + pEqn.flux();
        }
    }
}
else
{
    closedVolume = adjustPhi(phiHbyA, U, p);

    phiHbyA += fvc::interpolate(rho*(rAtU - rAU))*fvc::snGrad(p)*mesh.magSf();
    HbyA -= (rAU - rAtU)*fvc::grad(p);

    while (simple.correctNonOrthogonal())
    {
        fvScalarMatrix pEqn
        (
            fvc::div(phiHbyA)
          - fvm::laplacian(rhorAtU, p)
          ==
            fvOptions(psi, p, rho.name())
        );

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

// The incompressible form of the continuity error check is appropriate for
// steady-state compressible also.
#include "incompressible/continuityErrs.H"

// Explicitly relax pressure for momentum corrector
p.relax();

U = HbyA - rAtU*fvc::grad(p);
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
