// GPU offload (#3): k-epsilon on device. Validate (a) production GbyNu / nut, (b) the epsilonWallFunction
// near-wall values eps0 / G0, and (c) the eps & k transport-equation matrices (div - laplacian + reaction
// + SuSp(divU)) against the CPU k-epsilon.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fv_patch_field.cuh"
#include "fvc.cuh"
#include "fvm.cuh"
#include "fv_matrix_ops.cuh"
#include "near_wall_dist.cuh"
#include "nut_wall_function.cuh"
#include "k_epsilon.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "device_boundary.cuh"
#include "device_blas.cuh"
#include "device_kepsilon.cuh"
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace brae;

static scalar relMax(const std::vector<scalar>& a, const std::vector<scalar>& b) {
    scalar n = 0, d = 0; for (std::size_t i = 0; i < a.size(); ++i) { n = std::fmax(n, std::fabs(a[i]-b[i])); d = std::fmax(d, std::fabs(b[i])); }
    return d > 0 ? n / d : n;
}

int main(int argc, char** argv) {
    const std::string caseDir = argc > 1 ? argv[1] : "validation/pitzDaily";
    PrimitiveMesh m; m.read(caseDir + "/constant/polyMesh");
    FvGeometry g; g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells();
    const scalar nu = 1e-3, C1 = 1.44, sigmaK = 1.0, sigmaEps = 1.3, C3 = 0.0;

    GeometricField<vector> U; U.internal.resize(nC);
    for (label c = 0; c < nC; ++c) U.internal[c] = { 1.0 + 0.2*std::sin(0.01*c), 0.1*std::cos(0.013*c), 0.0 };
    for (const FvPatch& q : fvp) {
        if (q.type == "empty")      U.boundary.push_back(std::make_unique<EmptyPatchField<vector>>(q));
        else if (q.name == "inlet") U.boundary.push_back(std::make_unique<FixedValuePatchField<vector>>(q, true, vector{10,0,0}, std::vector<vector>{}));
        else if (q.type == "wall")  U.boundary.push_back(std::make_unique<NoSlipPatchField<vector>>(q));
        else                        U.boundary.push_back(std::make_unique<ZeroGradientPatchField<vector>>(q));
    }
    U.evaluateBoundary();
    auto mkScalar = [&](scalar a, scalar b, scalar f) { GeometricField<scalar> s; s.internal.resize(nC);
        for (label c = 0; c < nC; ++c) s.internal[c] = a + b*std::sin(f*c);
        for (const FvPatch& q : fvp) { if (q.type=="empty") s.boundary.push_back(std::make_unique<EmptyPatchField<scalar>>(q)); else s.boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(q)); }
        s.evaluateBoundary(); return s; };
    GeometricField<scalar> k = mkScalar(0.1, 0.05, 0.01), eps = mkScalar(1.0, 0.5, 0.013);
    const SurfaceScalarField phi = fvc::flux(U, m, g, fvp);

    // CPU references.
    const std::vector<tensor> gradU = fvc::gaussGrad(U, m, g, fvp);
    const std::vector<scalar> gByNuCpu = kepsilon::GbyNu(gradU);
    const std::vector<scalar> nutCpu = kepsilon::nut(k.internal, eps.internal);
    const std::vector<scalar> Gcpu = kepsilon::production(nutCpu, gByNuCpu);
    const std::vector<scalar> divU = fvc::div(phi, m, g, fvp);
    // wall functions eps0 / G0.
    const std::vector<std::vector<scalar>> yW = nearWallDist(m, g, fvp);
    const scalar kappa = 0.41, Cmu25 = std::pow(0.09,0.25), Cmu75 = std::pow(0.09,0.75);
    std::vector<label> nw(nC, 0);
    for (std::size_t pi = 0; pi < fvp.size(); ++pi) if (fvp[pi].type=="wall") for (label i=0;i<fvp[pi].size;++i) ++nw[fvp[pi].faceCells[i]];
    std::vector<scalar> eps0c(nC,0.0), G0c(nC,0.0);
    for (std::size_t pi = 0; pi < fvp.size(); ++pi) if (fvp[pi].type=="wall") {
        const std::vector<scalar> nutw = nutkWallFunction(fvp[pi], yW[pi], k.internal, nu);
        const std::vector<vector>& Uw = U.boundary[pi]->value();
        for (label i=0;i<fvp[pi].size;++i) { const label c=fvp[pi].faceCells[i]; const scalar w=1.0/nw[c], kc=k.internal[c];
            const scalar mg = mag((Uw[i]-U.internal[c])*fvp[pi].deltaCoeffs[i]);
            eps0c[c] += w*Cmu75*std::pow(kc,1.5)/(kappa*yW[pi][i]);
            G0c[c]   += w*(nutw[i]+nu)*mg*Cmu25*std::sqrt(kc)/(kappa*yW[pi][i]); }
    }
    // eps / k equation matrices (+ SuSp).
    FvScalarMatrix epsM = kepsilon::epsilonEqn(eps, k.internal, gByNuCpu, nutCpu, phi, nu, m, g, fvp);
    for (label c=0;c<nC;++c){ scalar sp=((2.0/3.0)*C1-C3)*divU[c]; epsM.diag[c]+=g.V()[c]*std::fmax(sp,0.0); epsM.source[c]-=g.V()[c]*std::fmin(sp,0.0)*eps.internal[c]; }
    FvScalarMatrix kM = kepsilon::kEqn(k, eps.internal, Gcpu, nutCpu, phi, nu, m, g, fvp);
    for (label c=0;c<nC;++c){ scalar sp=(2.0/3.0)*divU[c]; kM.diag[c]+=g.V()[c]*std::fmax(sp,0.0); kM.source[c]-=g.V()[c]*std::fmin(sp,0.0)*k.internal[c]; }

    // Device.
    const DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    const DeviceVectorBoundary dbU = buildDeviceVectorBoundary(U, fvp, g);
    const DeviceWallData wall = buildDeviceWallData(m, g, fvp, U);
    std::vector<scalar> ux(nC), uy(nC), uz(nC); for (label c=0;c<nC;++c){ux[c]=U.internal[c].x;uy[c]=U.internal[c].y;uz[c]=U.internal[c].z;}
    DeviceBuffer<scalar> dUx(ux), dUy(uy), dUz(uz), dk(k.internal), de(eps.internal);
    DeviceBuffer<scalar> gByNu; deviceGbyNu(dm, dbU, dUx, dUy, dUz, gByNu);
    DeviceBuffer<scalar> nutD; deviceNut(dk, de, nutD);
    DeviceBuffer<scalar> Gd; deviceHadamard(Gd, nutD, gByNu);
    DeviceBuffer<scalar> eps0d, G0d; deviceWallEpsG0(wall, dk, dUx, dUy, dUz, nu, eps0d, G0d);
    DeviceBuffer<scalar> phiB(([&]{std::vector<scalar> o; for(auto&a:phi.boundary)o.insert(o.end(),a.begin(),a.end());return o;}()));
    DeviceBuffer<scalar> divUd; deviceDiv(dm, DeviceBuffer<scalar>(phi.internal), phiB, divUd);
    // eps matrix
    std::vector<scalar> DepsC(nC), DkC(nC); for (label c=0;c<nC;++c){ DepsC[c]=nutCpu[c]/sigmaEps+nu; DkC[c]=nutCpu[c]/sigmaK+nu; }
    DeviceBuffer<scalar> Depsf; deviceInterpolate(dm, DeviceBuffer<scalar>(DepsC), Depsf);
    DeviceBuffer<scalar> eDiag,eUp,eLo,lDiag,lUp,lLo; deviceDivUpwindCoeffs(dm, DeviceBuffer<scalar>(phi.internal), eDiag,eUp,eLo); deviceLaplacianCoeffs(dm, Depsf, lDiag,lUp,lLo);
    deviceAxpy(-1.0,lDiag,eDiag); deviceAxpy(-1.0,lUp,eUp); deviceAxpy(-1.0,lLo,eLo);
    DeviceBuffer<scalar> eSrc(std::vector<scalar>(nC,0.0)); deviceEpsReaction(dm, de, dk, gByNu, divUd, eDiag, eSrc);
    // k matrix
    DeviceBuffer<scalar> Dkf; deviceInterpolate(dm, DeviceBuffer<scalar>(DkC), Dkf);
    DeviceBuffer<scalar> kDiag,kUp,kLo,l2D,l2U,l2L; deviceDivUpwindCoeffs(dm, DeviceBuffer<scalar>(phi.internal), kDiag,kUp,kLo); deviceLaplacianCoeffs(dm, Dkf, l2D,l2U,l2L);
    deviceAxpy(-1.0,l2D,kDiag); deviceAxpy(-1.0,l2U,kUp); deviceAxpy(-1.0,l2L,kLo);
    DeviceBuffer<scalar> kSrc(std::vector<scalar>(nC,0.0)); deviceKReaction(dm, dk, de, Gd, divUd, kDiag, kSrc);

    std::printf("GPU k-epsilon (nCells=%d):\n", nC);
    std::printf("  GbyNu %.2e  nut %.2e  production %.2e\n", relMax(gByNu.host(),gByNuCpu), relMax(nutD.host(),nutCpu), relMax(Gd.host(),Gcpu));
    std::printf("  wall eps0 %.2e   G0 %.2e\n", relMax(eps0d.host(),eps0c), relMax(G0d.host(),G0c));
    std::printf("  epsEqn diag %.2e up %.2e source %.2e\n", relMax(eDiag.host(),epsM.diag), relMax(eUp.host(),epsM.upper), relMax(eSrc.host(),epsM.source));
    std::printf("  kEqn   diag %.2e up %.2e source %.2e\n", relMax(kDiag.host(),kM.diag), relMax(kUp.host(),kM.upper), relMax(kSrc.host(),kM.source));
    const bool pass = relMax(gByNu.host(),gByNuCpu)<1e-12 && relMax(eps0d.host(),eps0c)<1e-12 && relMax(G0d.host(),G0c)<1e-12
                   && relMax(eDiag.host(),epsM.diag)<1e-12 && relMax(eSrc.host(),epsM.source)<1e-12
                   && relMax(kDiag.host(),kM.diag)<1e-12 && relMax(kSrc.host(),kM.source)<1e-12;
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
