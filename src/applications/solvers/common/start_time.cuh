#pragma once
// OF `startFrom` resolution, shared by the incompressible and compressible drivers.
//
// C6: gpuRhoSimpleFoam hardcoded `caseDir + "/0"` and never looked at startFrom at all, so a case with
// `startFrom latestTime` -- the standard way to CONTINUE a compressible run -- silently restarted from
// scratch. The run then converged perfectly to the right steady answer while having thrown away the
// restart, which is only harmless because the case was steady; the same input on a transient driver loses
// the history outright. The incompressible driver already resolved this; the logic lived inline there, so
// the compressible one could not share it. Extracted verbatim rather than reimplemented.
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

namespace brae {

// Resolve OF's startFrom against the case's numeric time directories. `startStr` is the controlDict
// startTime (the 'startTime' default, and the fallback when nothing matches). `probe` is a field that
// must exist for a directory to count as a field directory -- "U" for a flow solver -- so mesh-only
// directories written by snappyHexMesh are not mistaken for a restart point.
inline std::string resolveStartTime(
    const std::string& caseDir,
    const std::string& startFrom,
    const std::string& startStr,
    const char* probe = "U")
{
    if (startFrom != "latestTime" && startFrom != "firstTime") return startStr;

    namespace fs = std::filesystem;
    std::error_code ec;
    double best = 0.0;
    std::string bestName;
    for (const auto& e : fs::directory_iterator(caseDir, ec))
    {
        if (!e.is_directory()) continue;
        const std::string nm = e.path().filename().string();
        char* end = nullptr;
        const double t = std::strtod(nm.c_str(), &end);
        if (end == nm.c_str() || *end != '\0') continue;   // not a pure number -> skip constant/system/*.orig
        if (!(fs::exists(e.path() / probe) || fs::exists(e.path() / (std::string(probe) + ".gz")))) continue;
        if (bestName.empty() || (startFrom == "latestTime" ? t > best : t < best)) { best = t; bestName = nm; }
    }
    if (bestName.empty() || bestName == startStr) return startStr;

    std::fprintf(stderr, "brae: startFrom %s -> starting from time '%s'\n", startFrom.c_str(), bestName.c_str());
    return bestName;
}

}  // namespace brae
