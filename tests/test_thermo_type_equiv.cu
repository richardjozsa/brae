// heRhoThermo with perfectGas must parse to EXACTLY the same ThermoCoeffs as hePsiThermo.
//
// This is the right level for the claim. The solver-level version of this test was tried and abandoned:
// brae is not run-to-run bit-reproducible (GPU reduction ordering gives ~3e-9 on T after one iteration),
// so comparing two solver runs needs a statistical argument that 3 samples cannot settle. The actual
// claim -- "for perfectGas the two thermo types are the same arithmetic" -- is deterministic, so assert
// it deterministically: parse both dicts, compare every coefficient exactly.
//
// From OF source:
//   hePsiThermo::calculate  -> psi = mixture.psi(p,T);           rho follows as psi*p
//   heRhoThermo::calculate  -> psi = mixture.psi(p,T);  rho = mixture.rho(p,T)
//   perfectGas              -> rho = p/(R T), psi = 1/(R T)  =>  rho == psi*p exactly
//
// The second half is the point: with any OTHER equationOfState rho != psi*p, the two thermo types
// genuinely differ, and readThermoCoeffs must REFUSE heRhoThermo there. A test that only checked the
// perfectGas equivalence would pass just as well if the refusal were dropped.

#include "thermo_parse.cuh"
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

using namespace brae;

namespace {

int failures = 0;

// Writes a minimal case dir with the given thermoType and returns its path.
std::string writeCase(const std::string& dir, const std::string& type, const std::string& eos)
{
    std::filesystem::create_directories(dir + "/constant");
    std::filesystem::create_directories(dir + "/system");
    {
        std::ofstream f(dir + "/constant/thermophysicalProperties");
        f << "FoamFile { version 2.0; format ascii; class dictionary; object thermophysicalProperties; }\n"
          << "thermoType { type " << type << "; mixture pureMixture; transport const;\n"
          << "             thermo hConst; equationOfState " << eos << "; specie specie;\n"
          << "             energy sensibleEnthalpy; }\n"
          << "mixture { specie { molWeight 28.96; } thermodynamics { Cp 1005; Hf 0; }\n"
          << "          transport { mu 1.8e-05; Pr 0.7; } }\n";
    }
    {   // readThermoCoeffs also reads fvSolution for the bounds/relaxation
        std::ofstream f(dir + "/system/fvSolution");
        f << "FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }\n"
          << "solvers { }\n"
          << "SIMPLE { rhoMin 0.1; rhoMax 10; pMin 1000; }\n"
          << "relaxationFactors { fields { rho 0.7; } equations { h 0.7; } }\n";
    }
    return dir;
}

void eq(const char* what, scalar a, scalar b)
{
    if (a == b) return;
    std::printf("  FAIL %s differs: hePsiThermo %.17g vs heRhoThermo %.17g\n", what, a, b);
    failures++;
}

}   // namespace

int main()
{
    const std::string base = "/tmp/brae_thermo_type_equiv";
    std::filesystem::remove_all(base);

    const ThermoCoeffs psi = readThermoCoeffs(writeCase(base + "/psi", "hePsiThermo", "perfectGas"));
    const ThermoCoeffs rho = readThermoCoeffs(writeCase(base + "/rho", "heRhoThermo", "perfectGas"));

    eq("R", psi.R, rho.R);
    eq("Cp", psi.Cp, rho.Cp);
    eq("Hf", psi.Hf, rho.Hf);
    eq("Pr", psi.Pr, rho.Pr);
    eq("mu0", psi.mu0, rho.mu0);
    eq("As", psi.As, rho.As);
    eq("Ts", psi.Ts, rho.Ts);
    eq("Prt", psi.Prt, rho.Prt);
    eq("rhoMin", psi.rhoMin, rho.rhoMin);
    eq("rhoMax", psi.rhoMax, rho.rhoMax);
    eq("pMin", psi.pMin, rho.pMin);
    eq("relaxRho", psi.relaxRho, rho.relaxRho);
    if (psi.sutherland != rho.sutherland) { std::printf("  FAIL sutherland flag differs\n"); failures++; }
    if (psi.internalEnergy != rho.internalEnergy) { std::printf("  FAIL internalEnergy flag differs\n"); failures++; }
    if (failures == 0) std::printf("  OK   heRhoThermo/perfectGas parses identically to hePsiThermo\n");

    // NEGATIVE CONTROL: heRhoThermo with a non-perfectGas EOS must be REFUSED. Without this the
    // equivalence above would still pass if the refusal were removed, and brae would then run a
    // rho != psi*p equation of state through the perfect-gas path and converge to the wrong density.
    bool refused = false;
    try
    {
        readThermoCoeffs(writeCase(base + "/bad", "heRhoThermo", "rhoConst"));
    }
    catch (const std::exception&)
    {
        refused = true;
    }
    if (refused)
    {
        std::printf("  OK   heRhoThermo with a non-perfectGas EOS refused by name\n");
    }
    else
    {
        std::printf("  FAIL heRhoThermo + rhoConst was ACCEPTED -- rho != psi*p there, so the perfect-gas\n"
                    "       path would silently run the wrong density\n");
        failures++;
    }

    std::printf("thermo_type_equiv: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
