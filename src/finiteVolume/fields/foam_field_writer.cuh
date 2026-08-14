#pragma once
#include <map>
// brae::writeVolField, write a converged volume field in OpenFOAM's own structure. Unlike a raw text splice, this
// emits a fully RESOLVED field like OpenFOAM does: the internalField is the solved nonuniform list, and the
// boundaryField is written per mesh patch with an explicit `type` (+ value) entry, patch groups expanded, #include /
// #includeEtc / $macros / $internalField all resolved. So the file loads in any OpenFOAM reader (paraFoam, the
// built-in ParaView reader, postProcess, foamToVTK) exactly like a case OpenFOAM wrote itself.
#include "cf_types.cuh"
#include "foam_field_reader.cuh"   // readField -> FieldData/PatchFieldData (resolves includes/macros/$internalField)
#include "foam_dict.cuh"           // compileFoamRegex, isConstraintPatchType (same matching as buildField)
#include "fv_patch.cuh"            // FvPatch (name / type / inGroups)
#include <fstream>
#include <iomanip>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace brae {

inline void formatFoamValue(std::ostream& os, scalar v) { os << v; }
inline void formatFoamValue(std::ostream& os, const vector& v) { os << '(' << v.x << ' ' << v.y << ' ' << v.z << ')'; }
inline const char* foamListType(scalar) { return "List<scalar>"; }
inline const char* foamListType(const vector&) { return "List<vector>"; }
inline const char* foamClassName(scalar) { return "volScalarField"; }
inline const char* foamClassName(const vector&) { return "volVectorField"; }

// Write "uniform <v>;" or "nonuniform List<...> N (...);" for a boundary entry field.
template <typename T>
inline void writeFieldValue(
    std::ostream& os,
    bool uniform,
    const T& uval,
    const std::vector<T>& vals)
{
    if (uniform)
    {
        os << "uniform ";
        formatFoamValue(os, uval);
        os << ";\n";
    }
    else
    {
        os << "nonuniform " << foamListType(T{}) << " " << vals.size() << "(";
        for (const T& v : vals)
        {
            formatFoamValue(os, v);
            os << ' ';
        }
        os << ");\n";
    }
}

// One boundaryField patch entry in OpenFOAM structure: type, then the value entries the reader resolved (inletValue
// for inletOutlet/mixed, value for value-holding BCs). BCs that hold no value (zeroGradient/slip/symmetry/...) write
// just the type, matching OpenFOAM's output.
template <typename T>
inline void writePatchEntry(
    std::ostream& os,
    const std::string& name,
    const PatchFieldData<T>& d,
    const T* computed = nullptr,     // solver's boundary values for this patch, or null to echo the input
    std::size_t nComputed = 0)
{
    os << "    " << name << "\n    {\n";
    os << "        type            " << (d.type.empty() ? "zeroGradient" : d.type) << ";\n";
    if (d.hasInletValue)
    {
        os << "        inletValue      ";
        writeFieldValue(os, d.inletUniform, d.inletUniformValue, d.inletValues);
    }
    // fixedGradient's gradient is REQUIRED by OF's reader, not optional like value: omitting it made
    // OF abort with "Required entry 'gradient' : missing for patch ..." on any attempt to read brae's
    // output back. The gradient is an input that the solve does not change, so echo what was read.
    if (d.hasGradient)
    {
        os << "        gradient        ";
        writeFieldValue(os, d.gradientUniform, d.gradientUniformValue, d.gradientValues);
    }
    // flowRateInletVelocity: OF REQUIRES one of volumetricFlowRate/massFlowRate and aborts without it,
    // and so does brae ("has neither 'volumetricFlowRate' nor 'massFlowRate'"). Dropping it made brae's
    // own output unrestartable -- by brae OR by OpenFOAM -- which only surfaced once brae started
    // writing phi and a brae->brae restart was attempted at all. Same class as the `gradient` entry
    // above: an INPUT the solve does not change, so echo what was read. OF's own output writes the
    // Function1 in its "constant <v>" form.
    if (d.hasFlowRate)
    {
        os << "        " << (d.flowRateIsMass ? "massFlowRate" : "volumetricFlowRate")
           << "    constant " << d.flowRate << ";\n";
        if (d.rhoInlet >= 0) os << "        rhoInlet        " << d.rhoInlet << ";\n";
    }
    if (computed && nComputed)
    {
        // The SOLVED boundary values, not the ones the case was started from. Echoing the input made
        // every written boundaryField stale: a wall nut computed by the wall function was reported as
        // whatever 0/nut happened to say, so anything post-processing brae's output (yPlus, wall shear,
        // force decomposition, a restart) read a value the solve never used.
        // foamListType already IS "List<scalar>"/"List<vector>" -- wrapping it in another List<> emitted
        // "nonuniform List<List<scalar>>", which OF's reader rejects. Every scalar and vector field brae
        // wrote carried it, so no brae output could be restarted from or post-processed by OF; the
        // validation gates never caught it because they regex the internalField and never re-read the file.
        os << "        value           nonuniform " << foamListType(T{}) << " \n"
           << nComputed << "\n(\n";
        for (std::size_t i = 0; i < nComputed; ++i) { formatFoamValue(os, computed[i]); os << '\n'; }
        os << ")\n;\n";
    }
    else if (d.hasValue)
    {
        os << "        value           ";
        writeFieldValue(os, d.valueUniform, d.uniformValue, d.values);
    }
    os << "    }\n";
}

// Splice the solved internalField into a fully-resolved OpenFOAM field written for `patches`.
// A DERIVED field -- one brae computes but the case has no 0/ file for, e.g. rho. It still needs a
// template for the FoamFile header, but taking the template's identity with it is wrong: rho written from
// 0/T inherited `object T`, `dimensions [0 0 0 1 0 0 0]` (temperature), and T's boundary TYPES and VALUES,
// so the inlet density read back as 300 and a fixedGradient T wall became a density gradient of 20000.
// OF writes a derived field as `calculated` + the computed values on every non-constraint patch; this
// says so explicitly instead of inheriting.
struct DerivedFieldSpec
{
    const char* object = nullptr;       // FoamFile `object` entry, e.g. "rho"
    const char* dimensions = nullptr;   // full entry text, e.g. "dimensions      [1 -3 0 0 0 0 0];"
};

template <typename T, typename Patch>
inline void writeVolField(
    const std::string& origPath,
    const std::string& outPath,
    const std::vector<T>& values,
    const std::vector<Patch>& patches,
    int precision = 16,
    // Flat boundary values in patch order, EXCLUDING cyclic/cyclicAMI -- the same flattening
    // DeviceBoundary uses, so a solver buffer can be handed over unchanged. Empty -> echo the input.
    const std::vector<T>& computedBoundary = {},
    const DerivedFieldSpec* derived = nullptr)
{
    std::ifstream in(origPath);
    if (!in) throw std::runtime_error("writeVolField: cannot read " + origPath);
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

    // FoamFile header block (verbatim, it holds no directives) + the dimensions line.
    const std::size_t ff = text.find("FoamFile");
    const std::size_t hb = text.find('{', ff);
    int depth = 0;
    std::size_t he = hb;
    for (; he < text.size(); ++he)
    {
        if (text[he] == '{') ++depth;
        else if (text[he] == '}')
        {
            if (--depth == 0)
            {
                ++he;
                break;
            }
        }
    }
    std::string header = (ff == std::string::npos) ? "" : text.substr(0, he);
    if (derived && derived->object)
    {
        // Rewrite the template's `object <name>;` so the file identifies as what it IS. Some OF readers
        // key on it, and a mismatched one is confusing even where it is tolerated.
        const std::regex objRe("object\\s+[A-Za-z0-9_.:]+\\s*;");
        header = std::regex_replace(header, objRe, std::string("object      ") + derived->object + ";");
    }
    const std::size_t dk = text.find("dimensions", he == 0 ? 0 : he);
    const std::size_t ds = (dk == std::string::npos) ? std::string::npos : text.find(';', dk);
    const std::string dims = (derived && derived->dimensions)
        ? std::string(derived->dimensions)
        : ((ds == std::string::npos) ? "dimensions      [0 0 0 0 0 0 0];" : text.substr(dk, ds - dk + 1));

    // Resolved boundary (includes / #includeEtc / $macros / $internalField expanded); groups are matched per patch below.
    const FieldData<T> fd = readField<T>(origPath);

    std::ofstream out(outPath);
    if (!out) throw std::runtime_error("writeVolField: cannot write " + outPath);
    out << std::setprecision(precision);
    if (!header.empty()) out << header << "\n\n";
    out << dims << "\n\n";
    out << "internalField   nonuniform " << foamListType(T{}) << "\n" << values.size() << "\n(\n";
    for (const T& v : values)
    {
        formatFoamValue(out, v);
        out << '\n';
    }
    out << ")\n;\n\n";
    out << "boundaryField\n{\n";
    std::size_t bndOff = 0;   // index into computedBoundary (coupled patches excluded)
    for (const Patch& p : patches)   // per mesh patch, matched like buildField
    {
        const PatchFieldData<T>* d = nullptr;
        for (const auto& b : fd.boundary)   // pass 1: exact name
            if (b.name == p.name)
            {
                d = &b;
                break;
            }
        if (!d)
            for (const auto& b : fd.boundary)   // pass 2: group / regex
            {
                bool hit = false;
                for (const auto& g : p.inGroups)
                    if (b.name == g)
                    {
                        hit = true;
                        break;
                    }
                if (!hit)
                {
                    try
                    {
                        const std::regex re = compileFoamRegex(b.name);
                        if (std::regex_match(p.name, re)) hit = true;
                        else
                            for (const auto& g : p.inGroups)
                                if (std::regex_match(g, re))
                                {
                                    hit = true;
                                    break;
                                }
                    }
                    catch (const std::regex_error&) {}
                }
                if (hit) d = &b;
            }
        PatchFieldData<T> synth;                                                     // constraint patch (empty/cyclic/...) or safety
        if (!d || derived)
        {
            synth.name = p.name;
            // A derived field takes NO boundary type from the template -- constraint patches keep their own
            // (OF writes `empty` as `empty`), everything else is `calculated`, exactly as OF writes rho.
            synth.type = isConstraintPatchType(p.type)
                       ? p.type
                       : (derived ? "calculated" : "zeroGradient");
            d = &synth;
        }
        const bool coupled = (isCoupledInterfaceType(p.type));
        const bool haveComputed = !computedBoundary.empty() && !coupled && p.size > 0
                               && bndOff + static_cast<std::size_t>(p.size) <= computedBoundary.size()
                               && p.type != "empty";
        writePatchEntry(out, p.name, *d,
                        haveComputed ? &computedBoundary[bndOff] : nullptr,
                        haveComputed ? static_cast<std::size_t>(p.size) : 0);
        if (!coupled) bndOff += static_cast<std::size_t>(p.size);
    }
    out << "}\n\n\n// ************************************************************************* //\n";
}

// Write a surfaceScalarField (the face flux phi) in OpenFOAM structure, matching OF's own output: internalField = the
// internal-face flux, boundaryField per patch (`type calculated` + the face-flux value list, except `empty` patches
// which write only the type). phiBoundary is the FLAT boundary flux in patch order EXCLUDING cyclic/cyclicAMI patches
// (their flux lives on the interface, not in phiBnd) -- those coupled patches write just their constraint type (v1: no
// value). Loads in any OpenFOAM reader (paraFoam/postProcess) as the case's phi.
template <typename Patch>
inline void writeSurfaceField(
    const std::string& outPath,
    const std::vector<scalar>& phiInternal,   // nInternalFaces
    const std::vector<scalar>& phiBoundary,   // flat, patch order, EXCLUDING cyclic/cyclicAMI
    const std::vector<Patch>& patches,
    int precision = 16,
    // phi's dimensions differ by solver and OF writes the real ones: the incompressible solvers carry a
    // VOLUMETRIC flux [0 3 -1 0 0 0 0] (m3/s), the compressible ones a MASS flux [1 0 -1 0 0 0 0] (kg/s).
    // Defaulted to the volumetric form so the incompressible callers are unchanged. Getting this wrong
    // still loads, but every downstream tool that checks dimensions (postProcess, funkySetFields, a
    // restart into OF) then sees a field that claims to be something it is not.
    const std::string& dimensions = "[0 3 -1 0 0 0 0]",
    // Per-COUPLED-patch flux, keyed by patch name. cyclic/cyclicAMI/cyclicACMI faces are not in
    // phiBoundary at all (their flux lives on the interface object), so without this they were written
    // as a bare `type <patchType>;` with no value -- and a restart had nothing to resume from. OF writes
    // the values; see DeviceSimpleSolver::interfacePatchFlux for what the missing round-trip cost.
    // Absent/empty keeps the old value-less form, which is still right for a solver with no interface.
    const std::map<std::string, std::vector<scalar>>& coupledValues = {})
{
    std::ofstream out(outPath);
    if (!out) throw std::runtime_error("writeSurfaceField: cannot write " + outPath);
    out << std::setprecision(precision);
    out << "FoamFile\n{\n    version     2.0;\n    format      ascii;\n    class       surfaceScalarField;\n"
           "    location    \"" << outPath << "\";\n    object      phi;\n}\n\n";
    out << "dimensions      " << dimensions << ";\n\n";
    out << "internalField   nonuniform List<scalar> \n" << phiInternal.size() << "\n(\n";
    for (scalar v : phiInternal) out << v << '\n';
    out << ")\n;\n\n";
    out << "boundaryField\n{\n";
    std::size_t off = 0;
    for (const auto& p : patches)
    {
        out << "    " << p.name << "\n    {\n";
        if (isCoupledInterfaceType(p.type))   // coupled: flux held on the interface, not in phiBoundary
        {
            out << "        type            " << p.type << ";\n";
            const auto cv = coupledValues.find(p.name);
            if (cv != coupledValues.end() && cv->second.size() == static_cast<std::size_t>(p.size))
            {
                out << "        value           nonuniform List<scalar> \n" << p.size << "\n(\n";
                for (scalar v : cv->second) out << v << '\n';
                out << ")\n;\n";
            }
            out << "    }\n";
            continue;                                       // do NOT advance off (phiBoundary skips these)
        }
        const std::size_t n = static_cast<std::size_t>(p.size);
        if (p.type == "empty")
            out << "        type            empty;\n";
        else
        {
            out << "        type            calculated;\n        value           nonuniform List<scalar> \n" << n << "\n(\n";
            for (std::size_t i = 0; i < n; ++i) out << phiBoundary[off + i] << '\n';
            out << ")\n;\n";
        }
        off += n;
        out << "    }\n";
    }
    out << "}\n\n\n// ************************************************************************* //\n";
}

// Write a vol field with NO template (the CrankNicolson ddt0 fields, which have no 0/ prototype): a constructed OF header
// (class from T, the given object name + dimensions), the internalField, and a minimal boundaryField (constraint patches
// keep their type; every other patch is `calculated` with a zero uniform value). Loads in any OF reader; on a brae
// restart only the internalField is consumed (ddt0's boundary is unused by the pointwise recurrence).
template <typename T, typename Patch>
inline void writeVolFieldRaw(
    const std::string& outPath,
    const std::string& objectName,
    const std::string& dims,
    const std::vector<T>& values,
    const std::vector<Patch>& patches,
    int precision = 16,
    // Flat boundary values in patch order, EXCLUDING cyclic/cyclicAMI -- the same flattening
    // DeviceBoundary uses, so a solver buffer can be handed over unchanged. Empty -> echo the input.
    const std::vector<T>& computedBoundary = {})
{
    std::ofstream out(outPath);
    if (!out) throw std::runtime_error("writeVolFieldRaw: cannot write " + outPath);
    out << std::setprecision(precision);
    out << "FoamFile\n{\n    version     2.0;\n    format      ascii;\n    class       " << foamClassName(T{}) << ";\n"
           "    object      " << objectName << ";\n}\n\n";
    out << "dimensions      " << dims << ";\n\n";
    out << "internalField   nonuniform " << foamListType(T{}) << "\n" << values.size() << "\n(\n";
    for (const T& v : values) { formatFoamValue(out, v); out << '\n'; }
    out << ")\n;\n\n";
    out << "boundaryField\n{\n";
    for (const Patch& p : patches)
    {
        out << "    " << p.name << "\n    {\n";
        if (isConstraintPatchType(p.type))
            out << "        type            " << p.type << ";\n";
        else
        {
            out << "        type            calculated;\n        value           uniform ";
            formatFoamValue(out, T{});
            out << ";\n";
        }
        out << "    }\n";
    }
    out << "}\n\n\n// ************************************************************************* //\n";
}

} // namespace brae
