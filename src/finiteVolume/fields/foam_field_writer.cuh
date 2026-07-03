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

// Write "uniform <v>;" or "nonuniform List<...> N (...);" for a boundary entry field.
template <typename T>
inline void writeFieldValue(std::ostream& os, bool uniform, const T& uval, const std::vector<T>& vals) {
    if (uniform) { os << "uniform "; formatFoamValue(os, uval); os << ";\n"; }
    else {
        os << "nonuniform " << foamListType(T{}) << " " << vals.size() << "(";
        for (const T& v : vals) { formatFoamValue(os, v); os << ' '; }
        os << ");\n";
    }
}

// One boundaryField patch entry in OpenFOAM structure: type, then the value entries the reader resolved (inletValue
// for inletOutlet/mixed, value for value-holding BCs). BCs that hold no value (zeroGradient/slip/symmetry/...) write
// just the type, matching OpenFOAM's output.
template <typename T>
inline void writePatchEntry(std::ostream& os, const std::string& name, const PatchFieldData<T>& d) {
    os << "    " << name << "\n    {\n";
    os << "        type            " << (d.type.empty() ? "zeroGradient" : d.type) << ";\n";
    if (d.hasInletValue) { os << "        inletValue      "; writeFieldValue(os, d.inletUniform, d.inletUniformValue, d.inletValues); }
    if (d.hasValue)      { os << "        value           "; writeFieldValue(os, d.valueUniform, d.uniformValue,     d.values);      }
    os << "    }\n";
}

// Splice the solved internalField into a fully-resolved OpenFOAM field written for `patches`.
template <typename T, typename Patch>   // Patch = FvPatch (device solver) or PatchInfo (host solver); both carry name/type/inGroups
inline void writeVolField(const std::string& origPath, const std::string& outPath,
                          const std::vector<T>& values, const std::vector<Patch>& patches, int precision = 16) {
    std::ifstream in(origPath);
    if (!in) throw std::runtime_error("writeVolField: cannot read " + origPath);
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

    // FoamFile header block (verbatim, it holds no directives) + the dimensions line.
    const std::size_t ff = text.find("FoamFile"); const std::size_t hb = text.find('{', ff);
    int depth = 0; std::size_t he = hb;
    for (; he < text.size(); ++he) { if (text[he] == '{') ++depth; else if (text[he] == '}') { if (--depth == 0) { ++he; break; } } }
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
    for (const T& v : values) { formatFoamValue(out, v); out << '\n'; }
    out << ")\n;\n\n";
    out << "boundaryField\n{\n";
    for (const Patch& p : patches) {                                                // per mesh patch, matched like buildField
        const PatchFieldData<T>* d = nullptr;
        for (const auto& b : fd.boundary) if (b.name == p.name) { d = &b; break; }   // pass 1: exact name
        if (!d) for (const auto& b : fd.boundary) {                                  // pass 2: group / regex
            bool hit = false;
            for (const auto& g : p.inGroups) if (b.name == g) { hit = true; break; }
            if (!hit) { try { const std::regex re = compileFoamRegex(b.name);
                if (std::regex_match(p.name, re)) hit = true;
                else for (const auto& g : p.inGroups) if (std::regex_match(g, re)) { hit = true; break; }
            } catch (const std::regex_error&) {} }
            if (hit) d = &b;
        }
        PatchFieldData<T> synth;                                                     // constraint patch (empty/cyclic/...) or safety
        if (!d) { synth.name = p.name; synth.type = isConstraintPatchType(p.type) ? p.type : "zeroGradient"; d = &synth; }
        writePatchEntry(out, p.name, *d);
    }
    out << "}\n\n\n// ************************************************************************* //\n";
}

} // namespace brae
