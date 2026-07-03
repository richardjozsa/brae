// Instrumentation: dump kOmegaSST F1, F2, CDkOmega, S2 (+ the inputs k, omega, y and the gradients gradU,
// gradk, gradOmega) for a frozen field state, so cf can validate the SST blending/limiter formulas against OF
// using the SAME gradient inputs (isolates the pointwise formula logic from grad-reduction order). omega is
// derived from a converged k-epsilon state: epsilon = betaStar*k*omega -> omega = epsilon/(betaStar*k).
// Formulas verbatim from kOmegaSSTBase.C (OpenFOAM v2412): F1/F2/CDkOmega, S2 = 2*magSqr(symm(gradU)).
#include "fvCFD.H"
#include "wallDist.H"

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
    volScalarField nutIn(IOobject("nut", runTime.timeName(), mesh, IOobject::MUST_READ, IOobject::NO_WRITE), mesh);

    surfaceScalarField phi(IOobject("phi", runTime.timeName(), mesh, IOobject::MUST_READ, IOobject::NO_WRITE), mesh);

    const scalar betaStar = 0.09, alphaOmega2 = 0.856, nu = 1e-5;   // SST defaults; pitzDaily nu
    const scalar a1 = 0.31, b1 = 1.0, c1 = 10.0;                    // nut limiter (a1,b1) + Pk/GbyNu cap (c1)
    const scalar gamma1 = 5.0/9.0, gamma2 = 0.44, beta1 = 0.075, beta2 = 0.0828;   // omega-eqn blends

    // omega from the converged k-eps state (betaStar*k*omega = epsilon)
    volScalarField omega("omega", epsilon/(betaStar*k));

    const volScalarField& y = wallDist::New(mesh).y();   // meshWave + correctWalls (validated by ctest cell_wall_dist)

    volTensorField gradU(fvc::grad(U));
    volVectorField gradK(fvc::grad(k));
    volVectorField gradOmega(fvc::grad(omega));
    volScalarField GbyNu0f(gradU && devTwoSymm(gradU));   // production-by-nu (= k-eps GbyNu)
    volScalarField divUf(fvc::div(phi));                  // divU = fvc::div(fvc::absolute(phi,U)); phi absolute

    const scalarField& kf  = k.primitiveField();
    const scalarField& of  = omega.primitiveField();
    const scalarField& yf  = y.primitiveField();
    const tensorField& guf = gradU.primitiveField();
    const vectorField& gkf = gradK.primitiveField();
    const vectorField& gof = gradOmega.primitiveField();
    const scalarField& nutf  = nutIn.primitiveField();
    const scalarField& gbnu0 = GbyNu0f.primitiveField();
    const scalarField& divUp = divUf.primitiveField();
    const scalarField& Vol   = mesh.V();

    OFstream os(runTime.path()/"komega_sst.dat");
    os.precision(16);
    os << mesh.nCells() << nl;
    forAll(kf, c)
    {
        const scalar kk = kf[c], om = of[c], yy = yf[c];
        const tensor& g = guf[c];
        const scalar S2 = 2*magSqr(symm(g));
        const scalar CD = (2*alphaOmega2)*(gkf[c] & gof[c])/om;
        const scalar CDplus = Foam::max(CD, scalar(1.0e-10));
        const scalar arg1 = Foam::min
        (
            Foam::min
            (
                Foam::max((scalar(1)/betaStar)*Foam::sqrt(kk)/(om*yy), scalar(500)*nu/(yy*yy*om)),
                (4*alphaOmega2)*kk/(CDplus*yy*yy)
            ),
            scalar(10)
        );
        const scalar F1 = Foam::tanh(pow4(arg1));
        const scalar arg2 = Foam::min
        (
            Foam::max((scalar(2)/betaStar)*Foam::sqrt(kk)/(om*yy), scalar(500)*nu/(yy*yy*om)),
            scalar(100)
        );
        const scalar F2 = Foam::tanh(sqr(arg2));

        // C④ limiters
        const scalar GbyNu0 = gbnu0[c], nutInC = nutf[c];
        const scalar G = nutInC * GbyNu0;
        const scalar Pk = Foam::min(G, (c1*betaStar)*kk*om);
        const scalar denom = Foam::max(a1*om, b1*F2*Foam::sqrt(S2));
        const scalar nutNew = a1*kk/denom;                                   // correctNut
        const scalar GbyNuLim = Foam::min(GbyNu0, (c1/a1)*betaStar*om*denom);

        // C⑤ omega reaction (diag/source contributions), raw-scalar form of the omega block of correct().
        const scalar V    = Vol[c], divU = divUp[c];
        const scalar gamma = F1*(gamma1-gamma2)+gamma2, beta = F1*(beta1-beta2)+beta2;
        const scalar sp1 = (2.0/3.0)*gamma*divU, sp2 = (F1-1.0)*CD/om;
        const scalar reactDiag = V*(Foam::max(sp1,0.0) + beta*om + Foam::max(sp2,0.0));
        const scalar reactSrc  = V*gamma*GbyNuLim - V*Foam::min(sp1,0.0)*om - V*Foam::min(sp2,0.0)*om;

        // C⑥ k reaction: Pk(G) (Su) - SuSp((2/3)divU) - Sp(betaStar*omega); G = nut*GbyNu0 (raw, no wall override here).
        const scalar spk = (2.0/3.0)*divU;
        const scalar kReactDiag = V*(betaStar*om + Foam::max(spk,0.0));
        const scalar kReactSrc  = V*Pk - V*Foam::min(spk,0.0)*kk;

        os << kk << ' ' << om << ' ' << yy << ' '
           << g.xx()<<' '<<g.xy()<<' '<<g.xz()<<' '<<g.yx()<<' '<<g.yy()<<' '<<g.yz()<<' '<<g.zx()<<' '<<g.zy()<<' '<<g.zz()<<' '
           << gkf[c].x()<<' '<<gkf[c].y()<<' '<<gkf[c].z()<<' '
           << gof[c].x()<<' '<<gof[c].y()<<' '<<gof[c].z()<<' '
           << S2 << ' ' << CD << ' ' << F1 << ' ' << F2 << ' '
           << nutInC << ' ' << GbyNu0 << ' ' << nutNew << ' ' << Pk << ' ' << GbyNuLim << ' '
           << V << ' ' << divU << ' ' << reactDiag << ' ' << reactSrc << ' '
           << kReactDiag << ' ' << kReactSrc << nl;
    }
    Info << "komega_sst dumped (" << mesh.nCells() << " cells)" << endl;
    return 0;
}
