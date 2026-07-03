// Instrumentation: (a) solve a scalar convection-diffusion matrix div(phi,T)-laplacian(1,T)
// with the case's solver (PBiCGStab) -> transport.dat, and (b) dump the VECTOR momentum
// matrix div(phi,U)-laplacian(nu,U) (scalar diag/upper/lower + vector source/coeffs) ->
// momentum.dat. For cf to validate PBiCGStab and the vector matrix assembly.
#include "fvCFD.H"

int main(int argc, char *argv[])
{
    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"

    instantList times = runTime.times();
    runTime.setTime(times.last(), times.size() - 1);
    Info << "time " << runTime.timeName() << endl;

    volScalarField T(IOobject("T", runTime.timeName(), mesh, IOobject::MUST_READ, IOobject::NO_WRITE), mesh);
    volVectorField U(IOobject("U", runTime.timeName(), mesh, IOobject::MUST_READ, IOobject::NO_WRITE), mesh);
    surfaceScalarField phi(IOobject("phi", runTime.timeName(), mesh, IOobject::MUST_READ, IOobject::NO_WRITE), mesh);

    // (a) scalar convection-diffusion solve with PBiCGStab.
    {
        dimensionedScalar gamma("gamma", dimViscosity, scalar(1));   // value 1; dims to match div
        fvScalarMatrix M(fvm::div(phi, T) - fvm::laplacian(gamma, T));
        SolverPerformance<scalar> perf = M.solve();
        OFstream os(runTime.path()/"transport.dat");
        os.precision(16);
        os << mesh.nCells() << nl << perf.initialResidual() << ' ' << perf.finalResidual()
           << ' ' << perf.nIterations() << nl;
        forAll(T.primitiveField(), c) os << T[c] << nl;
        Info << "transport: init=" << perf.initialResidual() << " iters=" << perf.nIterations() << endl;
    }

    // (b) vector momentum matrix dump.
    {
        dimensionedScalar nu("nu", dimViscosity, scalar(1e-5));
        fvVectorMatrix M(fvm::div(phi, U) - fvm::laplacian(nu, U));
        OFstream os(runTime.path()/"momentum.dat");
        os.precision(16);
        os << mesh.nCells() << ' ' << mesh.nInternalFaces() << ' ' << mesh.boundary().size() << nl;
        forAll(M.diag(),  c) os << M.diag()[c]  << nl;
        forAll(M.upper(), f) os << M.upper()[f] << nl;
        const scalarField& low = M.lower();
        forAll(low,       f) os << low[f]        << nl;
        forAll(M.source(), c) { const vector& s = M.source()[c]; os << s.x() << ' ' << s.y() << ' ' << s.z() << nl; }
        forAll(mesh.boundary(), p) {
            const fvPatch& fp = mesh.boundary()[p];
            os << fp.name() << ' ' << fp.size() << nl;
            forAll(fp, i) {
                const vector& ic = M.internalCoeffs()[p][i];
                const vector& bc = M.boundaryCoeffs()[p][i];
                os << ic.x() << ' ' << ic.y() << ' ' << ic.z() << ' '
                   << bc.x() << ' ' << bc.y() << ' ' << bc.z() << nl;
            }
        }
        Info << "momentum matrix dumped" << endl;
    }

    // (c) fvMatrix A()/H() + explicit fvc::div(phi) and the face flux interpolate(U)&Sf.
    {
        dimensionedScalar nu("nu", dimViscosity, scalar(1e-5));
        fvVectorMatrix UEqn(fvm::div(phi, U) - fvm::laplacian(nu, U));
        volScalarField A(UEqn.A());
        volVectorField H(UEqn.H());
        volScalarField divPhi(fvc::div(phi));
        surfaceScalarField fluxU(fvc::interpolate(U) & mesh.Sf());

        OFstream os(runTime.path()/"ops.dat");
        os.precision(16);
        os << mesh.nCells() << ' ' << mesh.nInternalFaces() << ' ' << mesh.boundary().size() << nl;
        forAll(A.primitiveField(),      c) os << A[c] << nl;
        forAll(H.primitiveField(),      c) { const vector& h = H[c]; os << h.x() << ' ' << h.y() << ' ' << h.z() << nl; }
        forAll(divPhi.primitiveField(), c) os << divPhi[c] << nl;
        forAll(fluxU.primitiveField(),  f) os << fluxU[f] << nl;
        forAll(mesh.boundary(), p) {
            const fvsPatchScalarField& pf = fluxU.boundaryField()[p];
            os << mesh.boundary()[p].name() << ' ' << pf.size() << nl;
            forAll(pf, i) os << pf[i] << nl;
        }
        Info << "ops (A,H,div,flux) dumped" << endl;
    }

    // (d) pressure equation matrix laplacian(rAU, p), rAU = 1/UEqn.A().
    {
        dimensionedScalar nu("nu", dimViscosity, scalar(1e-5));
        fvVectorMatrix UEqn(fvm::div(phi, U) - fvm::laplacian(nu, U));
        volScalarField rAU("rAU", 1.0/UEqn.A());
        volScalarField p(IOobject("p", runTime.timeName(), mesh, IOobject::MUST_READ, IOobject::NO_WRITE), mesh);
        fvScalarMatrix pEqn(fvm::laplacian(rAU, p));

        OFstream os(runTime.path()/"peqn.dat");
        os.precision(16);
        os << mesh.nCells() << ' ' << mesh.nInternalFaces() << ' ' << mesh.boundary().size() << nl;
        forAll(pEqn.diag(),   c) os << pEqn.diag()[c]   << nl;
        forAll(pEqn.upper(),  f) os << pEqn.upper()[f]  << nl;
        const scalarField& low = pEqn.lower();
        forAll(low,           f) os << low[f]            << nl;
        forAll(pEqn.source(), c) os << pEqn.source()[c] << nl;
        forAll(mesh.boundary(), pp) {
            const fvPatch& fp = mesh.boundary()[pp];
            os << fp.name() << ' ' << fp.size() << nl;
            forAll(fp, i) os << pEqn.internalCoeffs()[pp][i] << ' ' << pEqn.boundaryCoeffs()[pp][i] << nl;
        }
        Info << "pEqn matrix dumped" << endl;
    }

    // (e) segregated momentum solve UEqn == 0 (component-wise).
    {
        dimensionedScalar nu("nu", dimViscosity, scalar(1e-5));
        fvVectorMatrix UEqn(fvm::div(phi, U) - fvm::laplacian(nu, U));
        SolverPerformance<vector> perf = UEqn.solve();
        OFstream os(runTime.path()/"usolve.dat");
        os.precision(16);
        os << mesh.nCells() << nl;
        forAll(U.primitiveField(), c) { const vector& u = U[c]; os << u.x() << ' ' << u.y() << ' ' << u.z() << nl; }
        Info << "U solve done (x iters=" << perf.nIterations() << ")" << endl;
    }
    return 0;
}
