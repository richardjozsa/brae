    // Solve the Momentum equation

    MRF.correctBoundaryVelocity(U);

    // Stage harness: U EXACTLY as UEqn.H assembles from -- after correctBoundaryVelocity and after the
    // boundary conditions have been evaluated, before any matrix is built. Without this, a brae-side
    // comparison of the momentum matrix also measures brae's flowRateInletVelocity (which sbMatched's
    // inlet uses, and whose value depends on which rho it is fed). Those are separate components and a
    // single number covering both cannot be attributed to either.
    if (runTime.timeIndex() == braeDumpIter())
    {
        volVectorField("stage_Uass", U).write();
        // rho EXACTLY as it stands when the momentum equation is assembled -- the field
        // flowRateInletVelocity::updateCoeffs looks up, so brae's inlet can be held against the same
        // density OpenFOAM held it against. It is the RELAXED solver rho from the previous iteration,
        // which is not thermo.rho() and not the face-cell values.
        volScalarField("stage_rhoU", rho).write();
        // phi EXACTLY as the momentum equation is convected by. From iteration 2 on this is the
        // CONSERVATIVE flux the previous pressure solve left, which is not what compressibleCreatePhi.H
        // would recompute from rho and U -- so a brae-side phi rebuilt at this point is a different field.
        surfaceScalarField("stage_phiU", phi).write();
    }

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

    if (runTime.timeIndex() == braeDumpIter())
    {   // cell volume, so the porosity's V*tr(Cd) contribution can be decomposed from D = V/rAU
        volScalarField VV
        (
            IOobject("stage_V", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
            mesh, dimensionedScalar(dimless, Zero), zeroGradientFvPatchScalarField::typeName
        );
        VV.primitiveFieldRef() = mesh.V();
        VV.write();
    }

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
        // thermo.mu(), the LAMINAR half of muEff with no turbulence in it -- the twin of
        // stage_alphaEffLam. muEff is INJECTED into the momentum gate so that its assembly number stays
        // attributable, which also means brae's own viscosity had never been compared to OpenFOAM's.
        // Its sibling alphaEff turned out to be six orders wrong behind exactly that injection.
        volScalarField("stage_muLam", thermo.mu()).write();
    }
