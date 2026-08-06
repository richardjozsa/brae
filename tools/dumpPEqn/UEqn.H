    // Solve the Momentum equation

    MRF.correctBoundaryVelocity(U);

    tmp<fvVectorMatrix> tUEqn
    (
        fvm::div(phi, U)
      + MRF.DDt(rho, U)
      + turbulence->divDevRhoReff(U)
     ==
        fvOptions(rho, U)
    );
    fvVectorMatrix& UEqn = tUEqn.ref();

    UEqn.relax();

    fvOptions.constrain(UEqn);

    // Stage harness: the MOMENTUM equation's boundary coefficients, per patch, AFTER relax() and
    // constrain() -- the same state brae's iC/bC are in. internalCoeffs goes to the adjacent cell's
    // diagonal, boundaryCoeffs to its source. With the diagonal proven right and the source proven wrong,
    // these are the whole remaining question.
    if (runTime.timeIndex() == braeDumpIter())
    {
        volVectorField UIC
        (
            IOobject("stage_UIC", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
            mesh, dimensionedVector(dimless, Zero), calculatedFvPatchVectorField::typeName
        );
        volVectorField UBC
        (
            IOobject("stage_UBC", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
            mesh, dimensionedVector(dimless, Zero), calculatedFvPatchVectorField::typeName
        );
        forAll(mesh.boundary(), pi)
        {
            UIC.boundaryFieldRef()[pi] = UEqn.internalCoeffs()[pi];
            UBC.boundaryFieldRef()[pi] = UEqn.boundaryCoeffs()[pi];
        }
        UIC.write();
        UBC.write();
        forAll(mesh.boundary(), pi)
        {
            const fvPatch& pp = mesh.boundary()[pi];
            if (!pp.size()) continue;
            vector si(Zero), sb(Zero);
            forAll(UEqn.internalCoeffs()[pi], i)
            {
                si += cmptMag(UEqn.internalCoeffs()[pi][i]);
                sb += cmptMag(UEqn.boundaryCoeffs()[pi][i]);
            }
            Info<< "  UEqn patch " << pp.name() << " (" << pp.size() << " faces)"
                << "  sum|iC| = " << si << "  sum|bC| = " << sb << endl;
        }
    }

    solve(UEqn == -fvc::grad(p));

    fvOptions.correct(U);

    // Stage harness: U straight out of the MOMENTUM PREDICTOR. UEqn.H() below uses THIS U, not the
    // initial field, so it is a stage in its own right and sits between rAU and HbyA in the chain.
    if (runTime.timeIndex() == braeDumpIter())
    {
        volVectorField("stage_Upred", U).write();
        volScalarField("stage_muEff", turbulence->muEff()).write();
    }
