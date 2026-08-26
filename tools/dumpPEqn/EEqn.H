{
    volScalarField& he = thermo.he();

    fvScalarMatrix EEqn
    (
        fvm::div(phi, he)
      + (
            he.name() == "e"
          ? fvc::div(phi, volScalarField("Ekp", 0.5*magSqr(U) + p/rho))
          : fvc::div(phi, volScalarField("K", 0.5*magSqr(U)))
        )
      - fvm::laplacian(turbulence->alphaEff(), he)
     ==
        fvOptions(rho, he)
    );

    if (MRF.active())
    {
        EEqn += fvc::div(MRF.phi(), p);
    }

    EEqn.relax();

    // Stage harness: the kinetic term and the assembled energy system at iteration 1.
    if (runTime.timeIndex() == braeDumpIter())
    {
        volScalarField Ekp
        (
            IOobject("stage_Ekp", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
            he.name() == "e" ? 0.5*magSqr(U) + p/rho : 0.5*magSqr(U)
        );
        Ekp.write();
        // The KINETIC TERM ITSELF, as OpenFOAM's own scheme produced it -- `div(phi,Ekp) bounded Gauss
        // upwind`, both halves. The field Ekp already agrees to 1e-16, so a disagreement in the assembled
        // source is this divergence and nothing else, and its boundary faces are where it would live.
        volScalarField keDiv
        (
            IOobject("stage_keDiv", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
            he.name() == "e"
          ? fvc::div(phi, volScalarField("Ekp", 0.5*magSqr(U) + p/rho))
          : fvc::div(phi, volScalarField("K", 0.5*magSqr(U)))
        );
        keDiv.write();
        // phi as the energy equation is convected by, so the comparison is not measuring a different flux.
        surfaceScalarField("stage_phiE", phi).write();
        volScalarField heDump
        (
            IOobject("stage_he", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
            he
        );
        heDump.write();
        volScalarField eD
        (
            IOobject("stage_eD", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
            mesh, dimensionedScalar(dimless, Zero), zeroGradientFvPatchScalarField::typeName
        );
        volScalarField eSrc(eD); eSrc.rename("stage_eSrc");
        eD.primitiveFieldRef() = EEqn.D();
        scalarField rhs(EEqn.source());
        forAll(mesh.boundary(), pj)
        {
            const labelUList& fc = mesh.boundary()[pj].faceCells();
            const scalarField& bcp = EEqn.boundaryCoeffs()[pj];
            forAll(fc, i) rhs[fc[i]] += bcp[i];
        }
        eSrc.primitiveFieldRef() = rhs;
        eD.write(); eSrc.write();
        volScalarField aEff
        (
            IOobject("stage_alphaEff", runTime.timeName(), mesh, IOobject::NO_READ, IOobject::NO_WRITE),
            turbulence->alphaEff()()
        );
        aEff.write();
        // thermo.alphahe() == CpByCpv*alpha, the LAMINAR half of alphaEff with no turbulence in it, so
        // the solver's own transport can be asserted against OpenFOAM's without also needing the model's
        // nut and alphat at this exact point.
        volScalarField aHe
        (
            IOobject("stage_alphaEffLam", runTime.timeName(), mesh, IOobject::NO_READ,
                     IOobject::NO_WRITE),
            thermo.alphahe()
        );
        aHe.write();
    }



    fvOptions.constrain(EEqn);

    EEqn.solve();

    fvOptions.correct(he);

    thermo.correct();
}
