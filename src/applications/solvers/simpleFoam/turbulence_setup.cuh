#pragma once
// Shared turbulence model + start-field setup for the steady (gpuSimpleFoam) and transient (gpuPimpleFoam) drivers.
// readTurbulenceModel: constant/turbulenceProperties RASModel -> ctl.turbulent/sst/sa/lm + coeffs. readTurbulenceFields:
// reads k/second/nut (or nuTilda/nut for SA, +ReThetat/gammaInt for LM), the nut wall function from the 0/nut BC,
// guards wall-function-on-non-wall patches, and applies the turbulent-inlet BCs. Bodies extracted verbatim from
// gpuSimpleFoam so both drivers stay identical. Call readTurbulenceModel BEFORE readTurbulenceFields (needs ctl.sst).
#include "simple_controls.cuh"
#include "foam_dict.cuh"
#include "foam_field_reader.cuh"
#include "geometric_field.cuh"
#include "fv_patch.cuh"
#include "kepsilon_coeffs.cuh"
#include "komega_sst_coeffs.cuh"
#include "spalart_coeffs.cuh"
#include "turbulent_inlet.cuh"
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace brae {

// constant/turbulenceProperties RASModel -> ctl turbulence flags + coeffs (ctl.turbulent must already be set).
inline void readTurbulenceModel(const FoamDict& turbProps, DeviceSimpleControls& ctl)
{
        if (ctl.turbulent)
        {
            const FoamDict* ras = turbProps.subDict("RAS");
            const std::string model = ras ? ras->wordOr("RASModel", "") : "";
            ctl.lm  = (model == "kOmegaSSTLM");                 // Langtry-Menter transition = kOmegaSST + gamma-ReThetat
            ctl.sst = (model == "kOmegaSST") || ctl.lm;
            ctl.sa  = (model == "SpalartAllmaras");
            const bool rke = (model == "realizableKE");
            if (model != "kEpsilon" && !rke && !ctl.sst && !ctl.sa)
                throw std::runtime_error("brae: unsupported RASModel '" + model + "' (kEpsilon, realizableKE, kOmegaSST, kOmegaSSTLM or SpalartAllmaras)");
            if (ctl.sa)
            {
                // Spalart-Allmaras: OF defaults (coeffs read from RAS.SpalartAllmarasCoeffs would override; not needed here).
                const SpalartAllmarasCoeffs& c = ctl.saCoeffs;
                std::printf("  SpalartAllmaras (OF defaults): sigmaNut=%.4g kappa=%.4g Cb1=%.4g Cb2=%.4g Cw1=%.4g Cw2=%.3g Cw3=%.3g Cv1=%.3g Cs=%.3g\n",
                            c.sigmaNut, c.kappa, c.Cb1, c.Cb2, c.Cw1(), c.Cw2, c.Cw3, c.Cv1, c.Cs);
            }
            else if (ctl.sst)
            {
                // kOmegaSST coefficients: OF turbulenceProperties RAS.kOmegaSSTCoeffs (absent keys keep OF defaults).
                readKOmegaSSTCoeffs(ras, ctl.ksstCoeffs);
                const KOmegaSSTCoeffs& c = ctl.ksstCoeffs;
                std::printf("  kOmegaSSTCoeffs: a1=%.4g betaStar=%.4g alphaK1=%.4g alphaOmega1=%.4g gamma1=%.4g beta1=%.4g\n",
                            c.a1, c.betaStar, c.alphaK1, c.alphaOmega1, c.gamma1, c.beta1);
            }
            else
            {
                // k-eps coefficients: RAS.kEpsilonCoeffs (kappa/E wall-function coeffs accepted if present).
                KEpsilonCoeffs& c = ctl.keCoeffs;
                if (rke)   // realizableKE: variable Cmu + strain production; OF defaults A0=4, C2=1.9, sigmak=1, sigmaEps=1.2
                {
                    c.realizable = true;
                    c.A0 = 4.0;
                    c.C2 = 1.9;
                    c.sigmaK = 1.0;
                    c.sigmaEps = 1.2;
                    if (const FoamDict* rkc = ras ? ras->subDict("realizableKECoeffs") : nullptr)
                    {
                        c.A0 = rkc->scalarOr("A0", c.A0);
                        c.C2 = rkc->scalarOr("C2", c.C2);
                        c.sigmaK = rkc->scalarOr("sigmak", c.sigmaK);
                        c.sigmaEps = rkc->scalarOr("sigmaEps", c.sigmaEps);
                        c.kappa = rkc->scalarOr("kappa", c.kappa);
                        c.E = rkc->scalarOr("E", c.E);
                    }
                    std::printf("  realizableKECoeffs: A0=%.4g C2=%.4g sigmak=%.4g sigmaEps=%.4g kappa=%.4g E=%.4g (variable Cmu)\n",
                                c.A0, c.C2, c.sigmaK, c.sigmaEps, c.kappa, c.E);
                }
                else
                {
                    const FoamDict* kec = ras ? ras->subDict("kEpsilonCoeffs") : nullptr;
                    if (kec)
                    {
                        c.Cmu = kec->scalarOr("Cmu", c.Cmu);
                        c.C1 = kec->scalarOr("C1", c.C1);
                        c.C2 = kec->scalarOr("C2", c.C2);
                        c.C3 = kec->scalarOr("C3", c.C3);
                        c.sigmaK = kec->scalarOr("sigmak", c.sigmaK);
                        c.sigmaEps = kec->scalarOr("sigmaEps", c.sigmaEps);
                        c.kappa = kec->scalarOr("kappa", c.kappa);
                        c.E = kec->scalarOr("E", c.E);
                    }
                    std::printf("  kEpsilonCoeffs%s: Cmu=%.4g C1=%.4g C2=%.4g C3=%.4g sigmak=%.4g sigmaEps=%.4g kappa=%.4g E=%.4g\n",
                                kec ? " (from dict)" : " (OF defaults)", c.Cmu, c.C1, c.C2, c.C3, c.sigmaK, c.sigmaEps, c.kappa, c.E);
                }
            }
        }
}

// The turbulence start fields, read from fieldDir with the nut wall function + turbulent-inlet BCs applied.
struct TurbulenceFields { GeometricField<scalar> k, eps, nut, ReThetat, gammaInt; };

inline TurbulenceFields readTurbulenceFields(const std::string& fieldDir, const std::vector<FvPatch>& fvp, label nC,
                                             DeviceSimpleControls& ctl, const std::string& secondName,
                                             const GeometricField<vector>& U)
{
        // Wall-function fidelity guard -- fail loud on a nut/epsilon/omega wall-function BC placed on a patch NOT typed
        // 'wall': brae gates the near-wall model on the geometric patch type, so the wall function would be SILENTLY
        // inert. Conservative on the patch match -- only errors when the BC patch resolves to a concrete non-'wall'
        // patch (group/regex names are skipped). NOTE: nutUSpalding/nutUBlended on a non-SA model are NO LONGER an
        // error -- brae now honours the velocity-based nut wall function on any RAS model (see setNutWall + the
        // NutWall dispatch in device_simple_foam.cuh), matching OpenFOAM.
        auto patchGeoType = [&](const std::string& nm) -> std::string {
            for (const auto& q : fvp) if (q.name == nm) return q.type;
            return "";
        };
        auto guardWallFn = [&](const FieldData<scalar>& fd, const std::string& field) {
            auto isWF = [](const std::string& t) {
                return t == "nutkWallFunction" || t == "nutUSpaldingWallFunction" || t == "nutLowReWallFunction"
                    || t == "nutUBlendedWallFunction" || t == "atmNutkWallFunction" || t == "epsilonWallFunction" || t == "omegaWallFunction";
            };
            for (const auto& pb : fd.boundary)
            {
                if (!isWF(pb.type)) continue;
                const std::string gt = patchGeoType(pb.name);
                if (!gt.empty() && gt != "wall")
                    throw std::runtime_error(field + " boundary '" + pb.name + "' uses " + pb.type + ", but the patch"
                        " is type '" + gt + "' (not 'wall'). brae applies the near-wall model only on 'wall' patches, so"
                        " it would be SILENTLY inert (no wall shear / near-wall constraint). Retype the patch as 'wall'"
                        " in constant/polyMesh/boundary.");
            }
        };
        // Pick the nut wall function from the 0/nut wall-patch BC TYPE (OpenFOAM does this per-BC, not by model):
        // nutUSpalding -> Spalding, nutUBlended -> Blended, else nutk. Warn once on nutLowRe (mapped to nutk: identical
        // only on a resolved y+<yPlusLam mesh). SA keeps its Spalding path regardless (ctl.sa short-circuits below).
        auto setNutWall = [&](const FieldData<scalar>& fd) {
            for (const auto& pb : fd.boundary)
            {
                const std::string gt = patchGeoType(pb.name);
                if (!gt.empty() && gt != "wall") continue;                 // only wall patches drive the choice
                if (pb.type == "nutUSpaldingWallFunction") { ctl.nutWall = NutWall::Spalding; }
                else if (pb.type == "nutUBlendedWallFunction") { ctl.nutWall = NutWall::Blended; }
                else if (pb.type == "atmNutkWallFunction")   // atmospheric rough-wall nut (k-based path + roughness z0)
                {
                    ctl.atmZ0 = pb.ablZ0;               // roughness length (from `z0` / $z0 include)
                    ctl.atmBoundNut = pb.atmBoundNut;  // clamp nut>=0 option
                    printf("  nut wall function: atmNutkWallFunction (rough, z0=%g, boundNut=%s) on %s per the BC\n",
                           (double)ctl.atmZ0, ctl.atmBoundNut ? "true" : "false", ctl.sst ? "kOmegaSST" : "kEpsilon");
                }
                else if (pb.type == "nutLowReWallFunction")
                    fprintf(stderr, "brae WARNING: nut boundary '%s' uses nutLowReWallFunction; brae applies "
                        "nutkWallFunction (log law). Identical only where y+<yPlusLam (resolved mesh).\n", pb.name.c_str());
            }
            if (!ctl.sa && ctl.nutWall != NutWall::Nutk)
                printf("  nut wall function: %s (velocity-based, honoured on %s per the BC)\n",
                       ctl.nutWall == NutWall::Spalding ? "nutUSpaldingWallFunction" : "nutUBlendedWallFunction",
                       ctl.sst ? "kOmegaSST" : "kEpsilon");
        };
        GeometricField<scalar> k, eps, nut, ReThetat, gammaInt;   // ReThetat/gammaInt: kOmegaSSTLM transition
        if (ctl.sa)   // Spalart-Allmaras (one-equation): nuTilda -> the "k" slot, nut. BCs (freestream + fixedValue-0 wall) need no inlet calc.
        {
            k   = buildField<scalar>(readField<scalar>(fieldDir + "/nuTilda"), fvp, nC);
            k.evaluateBoundary();
            const FieldData<scalar> nutFD = readField<scalar>(fieldDir + "/nut");
            guardWallFn(nutFD, "nut");
            nut = buildField<scalar>(nutFD, fvp, nC);
            nut.evaluateBoundary();
        }
        else if (ctl.turbulent)
        {
            const FieldData<scalar> kFD = readField<scalar>(fieldDir + "/k");
            const FieldData<scalar> sFD = readField<scalar>(fieldDir + "/" + secondName);
            k   = buildField<scalar>(kFD, fvp, nC);
            k.evaluateBoundary();
            eps = buildField<scalar>(sFD, fvp, nC);
            eps.evaluateBoundary();
            const FieldData<scalar> nutFD = readField<scalar>(fieldDir + "/nut");
            guardWallFn(nutFD, "nut");
            guardWallFn(sFD, secondName);
            setNutWall(nutFD);   // honour the BC-specified velocity-based nut wall function (nutUSpalding/nutUBlended)
            nut = buildField<scalar>(nutFD, fvp, nC);
            nut.evaluateBoundary();
            if (ctl.lm)   // kOmegaSSTLM transition fields
            {
                ReThetat = buildField<scalar>(readField<scalar>(fieldDir + "/ReThetat"), fvp, nC);
                ReThetat.evaluateBoundary();
                gammaInt = buildField<scalar>(readField<scalar>(fieldDir + "/gammaInt"), fvp, nC);
                gammaInt.evaluateBoundary();
                std::printf("  kOmegaSSTLM (Langtry-Menter transition): + ReThetat + gammaInt transport\n");
            }
            // turbulent-inlet BCs: inflow value computed from U (k) then from k (epsilon/omega).
            const scalar wCmu = ctl.sst ? ctl.ksstCoeffs.betaStar : ctl.keCoeffs.Cmu;
            applyTurbulentInletK(k, kFD, U, fvp);
            applyTurbulentInletSecond(eps, sFD, k, wCmu, fvp);
        }
    return { std::move(k), std::move(eps), std::move(nut), std::move(ReThetat), std::move(gammaInt) };
}

}  // namespace brae
