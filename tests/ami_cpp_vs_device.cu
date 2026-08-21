// The cyclicAMI CUDA path against the _cpp REFERENCE, one stage at a time.
//
// Each check pairs one device entry point with its host twin at IDENTICAL inputs. That is the property
// the reference exists for: brae's device AMI is a fused path, and until now a disagreement anywhere in
// it produced one number for the whole interface. On pipeCyclic 97% of the momentum residual sits on
// interface cells with the interior exactly zero, and four passes of reading the device code did not
// explain it -- because reading is not measuring.
//
// The inputs are the case's own converged fields, so neither side is fed the other's answer, and the
// bounds are 1e-12: these are the same arithmetic in two places, not two discretisations.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "ami_interface.cuh"
#include "cyclicAMI_cpp.cuh"
#include "device_mesh.cuh"
#include "device_ami.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace brae;

static int report(const char* name, const std::vector<scalar>& host, const std::vector<scalar>& dev,
                  scalar bound)
{
    if (host.size() != dev.size())
    {
        std::printf("  %-26s SIZE MISMATCH host %zu device %zu   FAIL\n", name, host.size(), dev.size());
        return 1;
    }
    scalar linf = 0;
    int worst = -1;
    for (std::size_t i = 0; i < host.size(); ++i)
    {
        const scalar r = std::fabs(dev[i] - host[i]) / std::fmax(std::fabs(host[i]), 1e-30);
        if (r > linf) { linf = r; worst = (int)i; }
    }
    const bool ok = linf < bound;
    std::printf("  %-26s L_inf rel %.3e   bound %.1e   %s\n", name, linf, bound, ok ? "ok" : "FAIL");
    if (!ok && worst >= 0)
        std::printf("      worst entry %d:  _cpp %.12e   device %.12e\n", worst, host[worst], dev[worst]);
    return ok ? 0 : 1;
}

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::printf("usage: %s <caseDir> <time>\n", argv[0]);
        return 2;
    }
    const std::string caseDir = argv[1], t = argv[2];

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells();

    const std::vector<AMIInterface> amis = buildAMIInterfaces(m, g, fvp);
    if (amis.empty()) { std::printf("SKIP: no cyclicAMI interface in %s\n", caseDir.c_str()); return 77; }

    auto rd = [&](const std::string& f) {
        GeometricField<scalar> x =
            buildField<scalar>(readField<scalar>(caseDir + "/" + t + "/" + f), fvp, nC);
        x.evaluateBoundary();
        return x;
    };
    GeometricField<vector> U =
        buildField<vector>(readField<vector>(caseDir + "/" + t + "/U"), fvp, nC);
    U.evaluateBoundary();
    GeometricField<scalar> p = rd("p"), nut = rd("nut");
    std::vector<scalar> nuEff(nC);
    for (label c = 0; c < nC; ++c) nuEff[c] = nut.internal[c] + 1e-6;

    const DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    DeviceAMI ami = buildDeviceAMI(amis);
    std::printf("ami_cpp_vs_device: %s/%s   %d interface(s), %d source faces\n",
                caseDir.c_str(), t.c_str(), (int)amis.size(), (int)ami.n);

    // The device packs every interface end to end; the reference works one at a time, so the host
    // results are concatenated in the same order.
    auto cat = [&](auto fn) {
        std::vector<scalar> out;
        for (const AMIInterface& a : amis) { const auto v = fn(a); out.insert(out.end(), v.begin(), v.end()); }
        return out;
    };

    // The state has to actually EXERCISE the stages. On a uniform axial initial field the interface
    // faces are azimuthal, so phi = U.Sf is ~0 and the momentum assembly's upwind split never fires --
    // the gate would pass on arithmetic it never ran. These ranges are printed so a degenerate input
    // cannot masquerade as agreement, and the flux check below refuses one outright.
    int rc = 0;
    std::vector<scalar> ux(nC), uy(nC), uz(nC);
    for (label c = 0; c < nC; ++c) { ux[c] = U.internal[c].x; uy[c] = U.internal[c].y; uz[c] = U.internal[c].z; }
    DeviceBuffer<scalar> dUx, dUy, dUz, dP, dNuEff, dV;
    dUx.copyFrom(ux); dUy.copyFrom(uy); dUz.copyFrom(uz);
    dP.copyFrom(p.internal); dNuEff.copyFrom(nuEff); dV.copyFrom(g.V());

    // ---- STAGE 1: scalar interpolate-to-source ---------------------------------------------------
    {
        DeviceBuffer<scalar> o;
        deviceAmiInterpolate(ami, dP, o);
        rc |= report("interpolate(p)", cat([&](const AMIInterface& a){
            return cpu::cyclicAMI::interpolate(a, p.internal); }), o.host(), 1e-12);
    }

    // ---- STAGE 2: rotated vector interpolate-to-source -------------------------------------------
    {
        DeviceBuffer<scalar> oX, oY, oZ;
        deviceAmiInterpolateVec(ami, dUx, dUy, dUz, oX, oY, oZ);
        for (int comp = 0; comp < 3; ++comp)
        {
            const std::vector<scalar> hv = cat([&](const AMIInterface& a){
                const std::vector<vector> v = cpu::cyclicAMI::interpolateVec(a, U.internal);
                std::vector<scalar> c(v.size());
                for (std::size_t i = 0; i < v.size(); ++i) c[i] = comp == 0 ? v[i].x : comp == 1 ? v[i].y : v[i].z;
                return c;
            });
            rc |= report(comp == 0 ? "interpolateVec(U).x" : comp == 1 ? "interpolateVec(U).y"
                                                                       : "interpolateVec(U).z",
                         hv, (comp == 0 ? oX : comp == 1 ? oY : oZ).host(), 1e-12);
        }
    }

    // ---- STAGE 3: the face value ------------------------------------------------------------------
    {
        DeviceBuffer<scalar> o;
        deviceAmiFaceValue(ami, dNuEff, o);
        rc |= report("faceValue(nuEff)", cat([&](const AMIInterface& a){
            return cpu::cyclicAMI::faceValue(a, nuEff); }), o.host(), 1e-12);
    }

    // ---- STAGE 4: the pressure (laplacian) interface assembly -------------------------------------
    {
        DeviceBuffer<scalar> diagD;
        diagD.copyFrom(std::vector<scalar>(nC, 0.0));
        deviceAmiAssembleLaplacian(ami, dNuEff, diagD, true);
        std::vector<scalar> diagH(nC, 0.0), ifH;
        for (const AMIInterface& a : amis)
        {
            cpu::cyclicAMI::State st;
            cpu::cyclicAMI::assembleLaplacian(a, nuEff, st, diagH, true);
            ifH.insert(ifH.end(), st.ifCoeff.begin(), st.ifCoeff.end());
        }
        rc |= report("assembleLaplacian ifCoeff", ifH,   ami.ifCoeff.host(), 1e-12);
        rc |= report("assembleLaplacian diag",    diagH, diagD.host(),       1e-12);
    }

    // ---- STAGE 5: the matrix action ---------------------------------------------------------------
    {
        std::vector<scalar> ApsiH(nC, 0.0);
        {
            std::size_t off = 0;
            const std::vector<scalar> ifAll = ami.ifCoeff.host();
            for (const AMIInterface& a : amis)
            {
                cpu::cyclicAMI::State st;
                st.ifCoeff.assign(ifAll.begin() + off, ifAll.begin() + off + a.ownCell.size());
                off += a.ownCell.size();
                cpu::cyclicAMI::amul(a, st, p.internal, ApsiH);
            }
        }
        DeviceBuffer<scalar> ApsiD;
        ApsiD.copyFrom(std::vector<scalar>(nC, 0.0));
        deviceAmiAmul(ami, dP, ApsiD);
        rc |= report("amul(p)", ApsiH, ApsiD.host(), 1e-12);
    }

    // ---- STAGE 6: the interface flux --------------------------------------------------------------
    {
        deviceAmiFlux(ami, dUx, dUy, dUz);
        rc |= report("flux(U)", cat([&](const AMIInterface& a){
            cpu::cyclicAMI::State st;
            cpu::cyclicAMI::flux(a, U.internal, st);
            return st.phi; }), ami.phi.host(), 1e-12);
    }

    // ---- STAGE 7: the MOMENTUM interface assembly -------------------------------------------------
    // The stage the open question points at: pipeCyclic's Uy momentum residual sits 97% on interface
    // cells. It runs after the flux above, because the upwind split needs the interface phi -- and both
    // sides are handed the SAME phi (the device's, which stage 6 just proved equals the reference's).
    const std::vector<scalar> phiIf = ami.phi.host();
    {
        DeviceBuffer<scalar> diagD;
        diagD.copyFrom(std::vector<scalar>(nC, 0.0));
        deviceAmiAssembleMomentum(ami, dNuEff, diagD);
        std::vector<scalar> diagH(nC, 0.0), ifH;
        std::size_t off = 0;
        for (const AMIInterface& a : amis)
        {
            const std::vector<scalar> ph(phiIf.begin() + off, phiIf.begin() + off + a.ownCell.size());
            off += a.ownCell.size();
            cpu::cyclicAMI::State st;
            cpu::cyclicAMI::assembleMomentum(a, nuEff, ph, st, diagH);
            ifH.insert(ifH.end(), st.ifCoeff.begin(), st.ifCoeff.end());
        }
        rc |= report("assembleMomentum ifCoeff", ifH,   ami.ifCoeff.host(), 1e-12);
        rc |= report("assembleMomentum diag",    diagH, diagD.host(),       1e-12);
    }

    // ---- STAGE 8: UEqn.H() ------------------------------------------------------------------------
    {
        DeviceBuffer<scalar> oX, oY, oZ;
        deviceAmiInterpolateVec(ami, dUx, dUy, dUz, oX, oY, oZ);
        const std::vector<scalar> UNx = oX.host();
        DeviceBuffer<scalar> Hd;
        Hd.copyFrom(std::vector<scalar>(nC, 0.0));
        deviceAmiAddH(ami, oX, dV, Hd);
        std::vector<scalar> Hh(nC, 0.0);
        std::size_t off = 0;
        const std::vector<scalar> ifAll = ami.ifCoeff.host();
        for (const AMIInterface& a : amis)
        {
            cpu::cyclicAMI::State st;
            st.ifCoeff.assign(ifAll.begin() + off, ifAll.begin() + off + a.ownCell.size());
            const std::vector<scalar> un(UNx.begin() + off, UNx.begin() + off + a.ownCell.size());
            off += a.ownCell.size();
            cpu::cyclicAMI::addH(a, st, un, g.V(), Hh);
        }
        rc |= report("addH(U.x)", Hh, Hd.host(), 1e-12);
    }

    // ---- STAGE 9: the interface's share of div(phi) ------------------------------------------------
    {
        DeviceBuffer<scalar> divD;
        divD.copyFrom(std::vector<scalar>(nC, 0.0));
        deviceAmiAddDiv(ami, dV, divD);
        std::vector<scalar> divH(nC, 0.0);
        std::size_t off = 0;
        for (const AMIInterface& a : amis)
        {
            cpu::cyclicAMI::State st;
            st.phi.assign(phiIf.begin() + off, phiIf.begin() + off + a.ownCell.size());
            off += a.ownCell.size();
            cpu::cyclicAMI::addDiv(a, st, g.V(), divH);
        }
        rc |= report("addDiv(phi)", divH, divD.host(), 1e-12);
    }

    {
        scalar mx = 0;
        for (scalar v : phiIf) mx = std::fmax(mx, std::fabs(v));
        std::printf("  interface flux |phi| max %.4e  (the upwind split is only exercised when this is "
                    "nonzero)\n", mx);
        if (!(mx > 1e-12))
        {
            std::printf("  FAIL: the interface flux is zero on this state, so assembleMomentum's "
                        "convective split was never exercised -- use a developed field, not a uniform one\n");
            rc = 1;
        }
    }

    std::printf("%s\n", rc == 0 ? "  ok:   every cyclicAMI CUDA stage matches the _cpp reference"
                                : "  FAIL: a cyclicAMI CUDA stage disagrees with the _cpp reference");
    return rc;
}
