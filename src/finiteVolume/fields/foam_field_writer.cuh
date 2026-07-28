#pragma once
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
    const PatchFieldData<T>& d)
{
    os << "    " << name << "\n    {\n";
    os << "        type            " << (d.type.empty() ? "zeroGradient" : d.type) << ";\n";
    if (d.hasInletValue)
    {
        os << "        inletValue      ";
        writeFieldValue(os, d.inletUniform, d.inletUniformValue, d.inletValues);
    }
    if (d.hasValue)
    {
        os << "        value           ";
        writeFieldValue(os, d.valueUniform, d.uniformValue, d.values);
    }
    os << "    }\n";
}

// Splice the solved internalField into a fully-resolved OpenFOAM field written for `patches`.
template <typename T, typename Patch>   // Patch = FvPatch (device solver) or PatchInfo (host solver); both carry name/type/inGroups
inline void writeVolField(
    const std::string& origPath,
    const std::string& outPath,
    const std::vector<T>& values,
    const std::vector<Patch>& patches,
    int precision = 16)
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
    const std::string header = (ff == std::string::npos) ? "" : text.substr(0, he);
    const std::size_t dk = text.find("dimensions", he == 0 ? 0 : he);
    const std::size_t ds = (dk == std::string::npos) ? std::string::npos : text.find(';', dk);
    const std::string dims = (ds == std::string::npos) ? "dimensions      [0 0 0 0 0 0 0];" : text.substr(dk, ds - dk + 1);

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
        if (!d)
        {
            synth.name = p.name;
            synth.type = isConstraintPatchType(p.type) ? p.type : "zeroGradient";
            d = &synth;
        }
        writePatchEntry(out, p.name, *d);
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
    int precision = 16)
{
    std::ofstream out(outPath);
    if (!out) throw std::runtime_error("writeSurfaceField: cannot write " + outPath);
    out << std::setprecision(precision);
    out << "FoamFile\n{\n    version     2.0;\n    format      ascii;\n    class       surfaceScalarField;\n"
           "    location    \"" << outPath << "\";\n    object      phi;\n}\n\n";
    out << "dimensions      [0 3 -1 0 0 0 0];\n\n";
    out << "internalField   nonuniform List<scalar> \n" << phiInternal.size() << "\n(\n";
    for (scalar v : phiInternal) out << v << '\n';
    out << ")\n;\n\n";
    out << "boundaryField\n{\n";
    std::size_t off = 0;
    for (const auto& p : patches)
    {
        out << "    " << p.name << "\n    {\n";
        if (p.type == "cyclic" || p.type == "cyclicAMI")   // coupled: flux held on the interface, not in phiBoundary
        {
            out << "        type            " << p.type << ";\n    }\n";
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
    int precision = 16)
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
