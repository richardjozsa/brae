// Focused parser coverage for the incompressible RAS turbulenceProperties model aliases.
#include "foam_dict.cuh"
#include "turbulence_setup.cuh"

#include <cstdio>
#include <exception>
#include <fstream>
#include <string>

using namespace brae;

namespace {
int failures = 0;

void check(bool ok, const char* what)
{
    if (!ok) { std::printf("  FAIL: %s\n", what); ++failures; }
    else       std::printf("  ok:   %s\n", what);
}

FoamDict dictFromString(const std::string& body)
{
    const std::string path = "test_turbulence_parser.tmpdict";
    {
        std::ofstream f(path);
        f << body;
    }
    FoamDict d = readDict(path);
    std::remove(path.c_str());
    return d;
}

bool readsModel(const std::string& body, const std::string& expectedModel)
{
    DeviceSimpleControls ctl;
    ctl.turbulent = true;
    readTurbulenceModel(dictFromString(body), ctl);
    if (expectedModel == "kOmegaSST") return ctl.sst && !ctl.lm && !ctl.sa;
    if (expectedModel == "kEpsilon") return !ctl.sst && !ctl.lm && !ctl.sa;
    return false;
}

std::string readError(const std::string& body)
{
    try
    {
        DeviceSimpleControls ctl;
        ctl.turbulent = true;
        readTurbulenceModel(dictFromString(body), ctl);
    }
    catch (const std::exception& e)
    {
        return e.what();
    }
    return {};
}
} // namespace

int main()
{
    std::printf("== turbulenceProperties model parser ==\n");

    // Existing Brae/OpenFOAM spelling remains valid.
    check(readsModel("simulationType RAS;\nRAS { RASModel kEpsilon; }\n", "kEpsilon"),
          "existing RASModel syntax selects kEpsilon");

    // OpenFOAM v2406 spelling is accepted without adapting the dictionary.
    check(readsModel("simulationType RAS;\nRAS { model kOmegaSST; }\n", "kOmegaSST"),
          "v2406 model syntax selects kOmegaSST");

    const std::string missing = readError("simulationType RAS;\nRAS { turbulence on; }\n");
    check(missing.find("unsupported RASModel ''") != std::string::npos,
          "missing model remains a clear error");

    const std::string unsupported = readError("simulationType RAS;\nRAS { model notARealModel; }\n");
    check(unsupported.find("unsupported RASModel 'notARealModel'") != std::string::npos,
          "unsupported model remains a clear error");

    check(readsModel("simulationType RAS;\nRAS { RASModel kOmegaSST; model kOmegaSST; }\n", "kOmegaSST"),
          "matching RASModel and model entries are accepted");

    const std::string conflict = readError(
        "simulationType RAS;\nRAS { RASModel kEpsilon; model kOmegaSST; }\n");
    check(conflict.find("conflicting RAS turbulence model entries") != std::string::npos
              && conflict.find("RASModel") != std::string::npos
              && conflict.find("model") != std::string::npos
              && conflict.find("kEpsilon") != std::string::npos
              && conflict.find("kOmegaSST") != std::string::npos,
          "conflicting model aliases fail with both keys and values named");

    std::printf(failures ? "== FAILED (%d) ==\n" : "== PASSED ==\n", failures);
    return failures ? 1 : 0;
}
