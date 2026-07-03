// Device cyclicAMI coupling unit test: build a synthetic AMI (weighted CSR stencil) and verify the device
// interpolate-to-source (scalar + rotated vector) matches a host reference to machine precision.
#include "device_ami.cuh"
#include "interface/cyclic_interface.cuh"   // rotationTensor
#include <cstdio>
#include <cmath>
#include <vector>

using namespace brae;

int main() {
    // synthetic AMI: src face 0 (cell 0) -> {cell3 w.6, cell4 w.4};  src face 1 (cell 1) -> {cell5 w1.0}
    AMIInterface a;
    a.ownCell    = {0, 1};
    a.srcOffset  = {0, 2, 3};
    a.nbrCell    = {3, 4, 5};
    a.weight     = {0.6, 0.4, 1.0};
    a.weightsSum = {1.0, 1.0};
    // per-src-face geometry (unused by this interp/Amul test but required to be present by buildDeviceAMI)
    a.magSf = {1.0, 1.0}; a.deltaCoeffs = {1.0, 1.0}; a.weights = {0.5, 0.5};
    a.Sf = {{1,0,0},{1,0,0}};
    a.translational = false;                               // exercise the rotational packing too
    a.forwardT = rotationTensor(vector{0,0,1}, M_PI/2);    // 90deg about z: (x,y,z) -> (-y,x,z)
    DeviceAMI d = buildDeviceAMI({a});

    const int nC = 6;
    std::vector<scalar> psi = {10, 11, 12, 13, 14, 15};
    std::vector<scalar> ux  = {0, 0, 0, 1.0, 2.0, 3.0};
    std::vector<scalar> uy  = {0, 0, 0, 0.5, 1.5, 2.5};
    std::vector<scalar> uz  = {0, 0, 0, 7.0, 8.0, 9.0};
    DeviceBuffer<scalar> dpsi, dux, duy, duz; dpsi.copyFrom(psi); dux.copyFrom(ux); duy.copyFrom(uy); duz.copyFrom(uz);

    int fail = 0;
    // scalar interpolate-to-source
    DeviceBuffer<scalar> sout; deviceAmiInterpolate(d, dpsi, sout);
    std::vector<scalar> sh = sout.host();
    scalar e0 = 0.6*psi[3] + 0.4*psi[4], e1 = 1.0*psi[5];
    if (std::fabs(sh[0]-e0) > 1e-13 || std::fabs(sh[1]-e1) > 1e-13) {
        std::printf("  scalar FAIL: got (%g,%g) exp (%g,%g)\n", sh[0], sh[1], e0, e1); ++fail; }

    // rotated vector interpolate-to-source: out = sum_k w*(R·U[nbr]); R(x,y,z)=(-y,x,z)
    DeviceBuffer<scalar> oX, oY, oZ; deviceAmiInterpolateVec(d, dux, duy, duz, oX, oY, oZ);
    auto hx = oX.host(), hy = oY.host(), hz = oZ.host();
    auto R = [&](int c, scalar& rx, scalar& ry, scalar& rz){ rx=-uy[c]; ry=ux[c]; rz=uz[c]; };
    for (int i = 0; i < 2; ++i) {
        scalar ex=0, ey=0, ez=0;
        for (label k = a.srcOffset[i]; k < a.srcOffset[i+1]; ++k) {
            scalar rx, ry, rz; R(a.nbrCell[k], rx, ry, rz);
            ex += a.weight[k]*rx; ey += a.weight[k]*ry; ez += a.weight[k]*rz;
        }
        if (std::fabs(hx[i]-ex)>1e-13 || std::fabs(hy[i]-ey)>1e-13 || std::fabs(hz[i]-ez)>1e-13) {
            std::printf("  vec FAIL i=%d: got (%g,%g,%g) exp (%g,%g,%g)\n", i,hx[i],hy[i],hz[i],ex,ey,ez); ++fail; }
    }
    // implicit Amul: Apsi[own[i]] += ifCoeff[i] * sum_k w[k]*psi[nbr[k]]
    std::vector<scalar> ifc = {2.0, -1.5}; d.ifCoeff.copyFrom(ifc);
    std::vector<scalar> Ap(nC, 0.0); DeviceBuffer<scalar> dAp; dAp.copyFrom(Ap);
    deviceAmiAmul(d, dpsi, dAp);
    auto ah = dAp.host();
    scalar a0 = ifc[0]*(0.6*psi[3]+0.4*psi[4]), a1 = ifc[1]*psi[5];   // -> cells 0,1
    if (std::fabs(ah[0]-a0) > 1e-12 || std::fabs(ah[1]-a1) > 1e-12) {
        std::printf("  amul FAIL: got (%g,%g) exp (%g,%g)\n", ah[0], ah[1], a0, a1); ++fail; }

    if (!fail) std::printf("test_ami_device PASS (scalar interp + rotated-vector interp + weighted Amul, %d nnz)\n", d.nnz);
    return fail ? 1 : 0;
}
