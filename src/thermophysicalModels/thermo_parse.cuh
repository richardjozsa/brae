#pragma once
// thermo_parse.cuh -- read constant/thermophysicalProperties into a ThermoCoeffs.
//
// Only the perfectGas + hConst + (sutherland | const) combination is supported today. Anything else is
// refused by name, with the supported set listed, rather than silently falling back to a default: a wrong
// equation of state does not announce itself in the residuals, it just gives the wrong answer. This is the
// same contract solver_dispatch.cuh applies to an unknown controlDict `application`.

#include "cf_types.cuh"
#include "foam_dict.cuh"
#include "thermo_types.cuh"
#include "thermo_model.cuh"   // thermoCv
#include <stdexcept>
#include <string>

namespace brae {

// Universal gas constant, OpenFOAM's specie::RR [J/(kmol K)]. molWeight is in kg/kmol, so R = RR/W.
inline constexpr scalar thermoRR = 8314.46261815324;

inline void thermoRequire(
    bool ok,
    const std::string& key,
    const std::string& got,
    const std::string& supported)
{
    if (ok) return;
    throw std::runtime_error(
        "brae: thermophysicalProperties thermoType." + key + " = '" + got
        + "' is not supported. brae supports: " + supported
        + ". (rhoSimpleFoam scope today is subsonic perfectGas.)");
}

// Reads the dictionary and fills ThermoCoeffs. Throws on anything outside the supported set.
inline ThermoCoeffs readThermoCoeffs(const std::string& caseDir)
{
    const FoamDict dict = readDict(caseDir + "/constant/thermophysicalProperties");
    ThermoCoeffs c;

    const FoamDict* tt = dict.subDict("thermoType");
    if (!tt)
    {
        throw std::runtime_error(
            "brae: constant/thermophysicalProperties has no thermoType dictionary.");
    }

    // Refuse the unsupported combinations up front, before any number is read.
    const std::string type = tt->wordOr("type", "");
    const std::string mixture = tt->wordOr("mixture", "");
    const std::string transport = tt->wordOr("transport", "");
    const std::string thermo = tt->wordOr("thermo", "");
    const std::string eos = tt->wordOr("equationOfState", "");
    const std::string energy = tt->wordOr("energy", "");

    thermoRequire(type == "hePsiThermo", "type", type, "hePsiThermo");
    thermoRequire(mixture == "pureMixture", "mixture", mixture, "pureMixture");
    thermoRequire(thermo == "hConst", "thermo", thermo, "hConst");
    thermoRequire(eos == "perfectGas", "equationOfState", eos, "perfectGas");
    thermoRequire(
        energy == "sensibleEnthalpy" || energy == "sensibleInternalEnergy",
        "energy",
        energy,
        "sensibleEnthalpy, sensibleInternalEnergy");
    c.internalEnergy = (energy == "sensibleInternalEnergy");
    thermoRequire(
        transport == "sutherland" || transport == "const",
        "transport",
        transport,
        "sutherland, const");

    c.sutherland = (transport == "sutherland");

    const FoamDict* mix = dict.subDict("mixture");
    if (!mix)
    {
        throw std::runtime_error(
            "brae: constant/thermophysicalProperties has no mixture dictionary.");
    }

    // specie: molWeight [kg/kmol] -> specific gas constant
    const FoamDict* specie = mix->subDict("specie");
    const scalar W = specie ? specie->scalarOr("molWeight", 28.9) : 28.9;
    if (W <= 0.0)
    {
        throw std::runtime_error("brae: mixture.specie.molWeight must be positive.");
    }
    c.R = thermoRR / W;

    // thermodynamics: hConst wants Cp and the heat of formation
    const FoamDict* thermoDict = mix->subDict("thermodynamics");
    if (thermoDict)
    {
        c.Cp = thermoDict->scalarOr("Cp", c.Cp);
        c.Hf = thermoDict->scalarOr("Hf", c.Hf);
    }

    // transport: sutherland wants (As, Ts), const wants (mu, Pr). Pr lives here in both cases.
    const FoamDict* transDict = mix->subDict("transport");
    if (transDict)
    {
        c.Pr = transDict->scalarOr("Pr", c.Pr);
        if (c.sutherland)
        {
            c.As = transDict->scalarOr("As", c.As);
            c.Ts = transDict->scalarOr("Ts", c.Ts);
        }
        else
        {
            c.mu0 = transDict->scalarOr("mu", c.mu0);
        }
    }

    if (c.Cp <= 0.0 || c.Pr <= 0.0)
    {
        throw std::runtime_error("brae: mixture Cp and Pr must be positive.");
    }

    // Cv = Cp - CpMCv, and perfectGas::CpMCv = R (OF HtoEthermo.H + perfectGasI.H). Guarded because a
    // molWeight/Cp pair giving Cp <= R is not a gas -- it would make gamma negative and the energy
    // equation quietly nonsense rather than obviously broken.
    if (thermoCv(c) <= 0.0)
    {
        throw std::runtime_error(
            "brae: Cv = Cp - R must be positive, but the given Cp and molWeight make it <= 0. "
            "That is not a gas: it would make gamma negative and the sutherland kappa nonsense.");
    }

    // Bounds are a solver control, not a thermo property, so they come from fvSolution when present.
    // Absent, the ThermoCoeffs defaults stand.
    const FoamDict fvSolution = readDict(caseDir + "/system/fvSolution");
    const FoamDict* simple = fvSolution.subDict("SIMPLE");
    if (simple)
    {
        c.rhoMin = simple->scalarOr("rhoMin", c.rhoMin);
        c.rhoMax = simple->scalarOr("rhoMax", c.rhoMax);
        c.pMin = simple->scalarOr("pMin", c.pMin);
    }

    // rho.relax() factor lives under relaxationFactors.fields.rho, as in OF.
    const FoamDict* relax = fvSolution.subDict("relaxationFactors");
    if (relax)
    {
        const FoamDict* fields = relax->subDict("fields");
        if (fields) c.relaxRho = fields->scalarOr("rho", c.relaxRho);
    }

    // Prt, exactly where OF looks for it: the turbulence model's own coeffs dict
    // (EddyDiffusivity::correctNut -> Prt_.readIfPresent(this->coeffDict())). Absent, OF's 1.0 stands.
    // OF names the dict after the model, e.g. RAS { RASModel kOmegaSST; kOmegaSSTCoeffs { Prt 0.85; } }.
    try
    {
        const FoamDict turbProps = readDict(caseDir + "/constant/turbulenceProperties");
        for (const char* sub : {"RAS", "LES"})
        {
            const FoamDict* d = turbProps.subDict(sub);
            if (!d) continue;
            const std::string model = d->wordOr(std::string(sub) + "Model", "");
            const FoamDict* coeffs = model.empty() ? nullptr : d->subDict(model + "Coeffs");
            if (coeffs) c.Prt = coeffs->scalarOr("Prt", c.Prt);
        }
    }
    catch (const std::exception&)
    {
        // No turbulenceProperties (a laminar case may omit it): alphat is zero anyway, so Prt is unused.
    }

    return c;
}

} // namespace brae
