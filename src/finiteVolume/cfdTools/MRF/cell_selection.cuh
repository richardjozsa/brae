#pragma once
// Cell selection for fvOptions/MRF -- OF src/fvOptions/cellSetOption.
//
// WHY IT LIVES HERE AND NOT IN fv_options.cuh. In OpenFOAM, selection is NOT an individual source's
// business: every fv::option derives from cellSetOption, which owns
//
//     enum selectionModeType { smAll, smCellSet, smCellZone, smPoints, smGeometric }   cellSetOption.H:175
//
// and resolves it once. A new source type inherits all of them for free. brae had the same shape for
// cellZone only, resolved inside readFvOptions, so `selectionMode cellSet` was unsupported for EVERY
// source at once -- simpleFoam/turbineSiting refused with "uses selectionMode cellSet 'actuationDisk1'
// with no matching cellZone", even though the actuationDiskSource[Froude] it needed was already
// implemented. Fixing it inside the actuation-disk branch would have fixed one source; fixing it here
// fixes explicitPorositySource, meanVelocityForce, limitVelocity, rotorDisk and everything added later,
// and lets MRF (which already reads cellZones) share the resolver.
//
// SCOPE. `all`, `cellZone` and `cellSet` are implemented. `points` and `geometric` are NOT, and are
// reported by name rather than silently falling back to `all` -- selecting the whole mesh instead of a
// turbine disk is a plausible-looking wrong answer, which is the failure mode this codebase refuses.

#include "cf_types.cuh"
#include "foam_token_reader.cuh"
#include "mrf_read.cuh"   // readCellZones + foamFormat, the cellZone half of the same job
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace brae {

// One cellSet: constant/polyMesh/sets/<name>. OF writes it as a labelList of CELL indices -- the same
// payload cellZones carries per zone, in its own file. Empty if absent.
inline std::vector<label> readCellSet(const std::string& polyMeshDir, const std::string& name)
{
    const std::string path = polyMeshDir + "/sets/" + name;
    {
        std::ifstream f(path);
        std::ifstream g(path + ".gz");
        if (!f.good() && !g.good()) return {};
    }
    std::vector<label> cells;
    TokenStream ts(path);
    // The file is a bare labelList: <n> ( l0 l1 ... ). Skip anything before the first count so the
    // FoamFile header does not have to be modelled.
    while (!ts.eof())
    {
        const std::string t = ts.next();
        if (t.empty()) break;
        if (t == "(")            // a list opened without a preceding count we recognised
        {
            while (!ts.eof())
            {
                const std::string v = ts.next();
                if (v == ")" || v.empty()) break;
                cells.push_back(static_cast<label>(std::strtol(v.c_str(), nullptr, 10)));
            }
            break;
        }
        char* end = nullptr;
        const long n = std::strtol(t.c_str(), &end, 10);
        if (end != t.c_str() && *end == '\0' && n >= 0 && !ts.eof())
        {
            const std::string nxt = ts.next();
            if (nxt == "(")
            {
                cells.reserve(static_cast<std::size_t>(n));
                for (long i = 0; i < n && !ts.eof(); ++i)
                {
                    const std::string v = ts.next();
                    if (v == ")" || v.empty()) break;
                    cells.push_back(static_cast<label>(std::strtol(v.c_str(), nullptr, 10)));
                }
                break;
            }
        }
    }
    return cells;
}

// The outcome of resolving one source's selectionMode. `ok == false` carries the reason, so the caller
// reports it by name instead of running with the wrong cells.
struct CellSelection
{
    std::vector<label> cells;          // empty with ok == true means "all cells"
    bool               all = false;
    bool               ok  = true;
    std::string        reason;
};

// OF cellSetOption::setSelection. `zones` is readCellZones()'s map, passed in so a caller that already
// has it does not re-read the file.
inline CellSelection resolveCellSelection(
    const std::string& polyMeshDir,
    const std::string& selectionMode,
    const std::string& setOrZoneName,
    const std::map<std::string, std::vector<label>>& zones)
{
    CellSelection s;
    if (selectionMode.empty() || selectionMode == "all")
    {
        s.all = true;
        return s;
    }
    if (selectionMode == "cellZone")
    {
        const auto it = zones.find(setOrZoneName);
        if (it == zones.end())
        {
            s.ok = false;
            s.reason = "selectionMode cellZone '" + setOrZoneName + "' is not in constant/polyMesh/cellZones";
            return s;
        }
        s.cells = it->second;
        return s;
    }
    if (selectionMode == "cellSet")
    {
        s.cells = readCellSet(polyMeshDir, setOrZoneName);
        if (s.cells.empty())
        {
            s.ok = false;
            s.reason = "selectionMode cellSet '" + setOrZoneName +
                       "' could not be read from constant/polyMesh/sets/" + setOrZoneName +
                       " (run topoSet first, or check the name)";
        }
        return s;
    }
    // points / geometric: NOT implemented. Reported, never silently widened to `all`.
    s.ok = false;
    s.reason = "selectionMode '" + selectionMode + "' is not implemented (brae supports all, cellZone, "
               "cellSet); falling back to all cells would apply the source to the whole mesh";
    return s;
}

}   // namespace brae
