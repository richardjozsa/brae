#pragma once
// Reads an OpenFOAM field file: dimensions / internalField (uniform|nonuniform) /
// boundaryField { patch { type; value; ... } }. Templated on the value type (scalar/vector).
// ASCII for now; binary field values land with the binary field reader (later increment).
#include "cf_types.cuh"
#include "foam_token_reader.cuh"
#include <string>
#include <vector>
#include <filesystem>

namespace brae {

template <typename T> inline T readFoamValue(TokenStream& ts);
template <> inline scalar readFoamValue<scalar>(TokenStream& ts) { return ts.nextScalar(); }
template <> inline vector readFoamValue<vector>(TokenStream& ts)
{
    ts.expect("(");
    vector v{ts.nextScalar(), ts.nextScalar(), ts.nextScalar()};
    ts.expect(")");
    return v;
}

// Skip an unhandled dict entry's tokens up to (not including) its terminating ';', paren-aware so nested "(...)" lists
// / tables / Function1 entries are consumed whole. initialDepth accounts for a leading '(' the caller already read.
// Leaves the ';' in place for the caller's ts.expect(";"). Single source for the reader's 3 table/skip fallbacks.
inline void skipToSemicolon(TokenStream& ts, int initialDepth = 0)
{
    int depth = initialDepth;
    while (!(depth == 0 && ts.peek() == ";"))
    {
        const std::string s = ts.next();
        if (s == "(") ++depth;
        else if (s == ")") --depth;
    }
}

template <typename T>
struct PatchFieldData
{
    std::string    name;
    std::string    type;
    bool           hasValue     = false;
    bool           valueUniform = false;
    T              uniformValue{};
    std::vector<T> values;
    // inletOutlet (and similar mixed BCs): the inflow value. Used as the device refValue.
    bool           hasInletValue   = false;
    bool           inletUniform    = false;
    T              inletUniformValue{};
    std::vector<T> inletValues;
    // turbulent-inlet BCs: k = 1.5*(intensity*|U|)^2; eps = Cmu^0.75 k^1.5/mixingLength; omega = sqrt(k)/(Cmu^0.25 mixingLength).
    scalar         intensity    = 0;
    scalar         mixingLength = 0;
    // surfaceNormalFixedValue / uniformNormalFixedValue: SCALAR refValue; the BC builds U_b = refValue * face_normal.
    bool                hasNormalRef     = false;
    bool                normalRefUniform = false;
    scalar              normalRefUniformValue = 0;
    std::vector<scalar> normalRefValues;
    // timeVaryingMappedFixedValue: the external boundaryData points + values (read here; the BC maps them to the faces).
    bool                hasMapData = false;
    std::vector<vector> mapPoints;
    std::vector<T>      mapValues;
    // atmBoundaryLayerInlet{Velocity,K,Epsilon,Omega}: log-law ABL profile params (the BC evaluates the profile per
    // face from Cf). u* = kappa*|Uref|/ln((Zref+z0)/z0); U(z)=(u*/kappa)ln((z-d+z0)/z0)*flowDir; k=u*^2/sqrt(Cmu);
    // eps=u*^3/(kappa(z-d+z0)); omega=u*/(sqrt(Cmu)kappa(z-d+z0)); z = Cf.zDir.
    bool   hasABL = false;
    scalar ablUref = 0, ablZref = 0, ablZ0 = 0.1, ablD = 0, ablKappa = 0.41, ablCmu = 0.09;
    vector ablFlowDir{1, 0, 0}, ablZDir{0, 0, 1};
};

template <typename T>
struct FieldData
{
    bool                          internalUniform = true;
    T                             internalUniformValue{};
    std::vector<T>                internalField;   // when nonuniform
    std::vector<PatchFieldData<T>> boundary;
};

// Read "uniform <v>" or "nonuniform List<...> N ( ... )".
template <typename T>
inline void readUniformOrList(
    TokenStream& ts,
    bool& uniform,
    T& uval,
    std::vector<T>& vals)
{
    const std::string mode = ts.next();
    if (mode == "uniform")
    {
        uniform = true;
        uval = readFoamValue<T>(ts);
    }
    else   // nonuniform
    {
        uniform = false;
        ts.next();                 // List<scalar> / List<vector>
        const label n = ts.nextLabel();
        ts.expect("(");
        vals.resize(n);
        for (label i = 0; i < n; ++i)
            vals[i] = readFoamValue<T>(ts);
        ts.expect(")");
    }
}

// Read "uniform <v>" / "nonuniform List<...>" OR the OF self-reference "$internalField" (copy the internalField
// entry). Used by value / inletValue / freestreamValue, any of which may be written as $internalField.
template <typename T>
inline void readValueOrInternal(
    TokenStream& ts,
    const FieldData<T>& fd,
    bool& uniform,
    T& uval,
    std::vector<T>& vals)
{
    if (ts.peek() == "$internalField")
    {
        ts.next();
        uniform = fd.internalUniform;
        uval = fd.internalUniformValue;
        vals = fd.internalField;
    }
    else readUniformOrList(ts, uniform, uval, vals);
}

// timeVaryingMappedFixedValue boundaryData
// Read an OF boundaryData list file: [comments] N ( v1 v2 ... ). Comments are stripped by the tokenizer.
template <typename V>
inline std::vector<V> readBoundaryDataList(const std::string& file)
{
    TokenStream ts(file);
    const label n = ts.nextLabel();
    ts.expect("(");
    std::vector<V> vals(n);
    for (label i = 0; i < n; ++i)
        vals[i] = readFoamValue<V>(ts);
    return vals;   // closing ')' not required by the mapper
}
// Read constant/boundaryData/<patch>/{points, <earliestTime>/<field>}. Steady simpleFoam: the earliest boundaryData
// time IS the (constant) profile (no time interpolation). The BC then maps these points->faces (nearest).
template <typename T>
inline void readTimeVaryingMapped(
    const std::string& fieldPath,
    const std::string& patchName,
    PatchFieldData<T>& p)
{
    namespace fs = std::filesystem;
    const std::size_t s1 = fieldPath.rfind('/');
    const std::string field   = fieldPath.substr(s1 + 1);
    const std::string timeP   = fieldPath.substr(0, s1);
    const std::string caseDir = timeP.substr(0, timeP.rfind('/'));
    const std::string bd = caseDir + "/constant/boundaryData/" + patchName;
    std::string best;
    double bestT = 1e300;
    for (const auto& e : fs::directory_iterator(bd))
    {
        if (!e.is_directory()) continue;
        try
        {
            const double t = std::stod(e.path().filename().string());
            if (t < bestT)
            {
                bestT = t;
                best = e.path().filename().string();
            }
        }
        catch (...) {}
    }
    if (best.empty()) return;
    p.mapPoints = readBoundaryDataList<vector>(bd + "/points");
    p.mapValues = readBoundaryDataList<T>(bd + "/" + best + "/" + field);
    p.hasMapData = (p.mapPoints.size() == p.mapValues.size() && !p.mapPoints.empty());
}

template <typename T>
inline FieldData<T> readField(const std::string& path)
{
    TokenStream ts(path, /*expandVars=*/true);   // expand in-file $macros (e.g. Uinlet (0 1 0); ... $Uinlet)
    FieldData<T> fd;

    while (!ts.eof())
    {
        const std::string t = ts.next();
        if (t == "internalField")
        {
            readUniformOrList(ts, fd.internalUniform, fd.internalUniformValue, fd.internalField);
            ts.expect(";");
        }
        else if (t == "boundaryField")
        {
            ts.expect("{");
            while (ts.peek() != "}")
            {
                PatchFieldData<T> p;
                p.name = ts.next();
                ts.expect("{");
                while (ts.peek() != "}")
                {
                    const std::string key = ts.next();
                    if (key == ";") continue;                // stray ';' left by a subdict-macro ($intakeType1;) expansion
                    if (key == "type")
                    {
                        p.type = ts.next();
                        ts.expect(";");
                        if (p.type.rfind("atmBoundaryLayer", 0) == 0) p.hasABL = true;
                    }
                    // atmBoundaryLayerInlet* params (from the case's include/ABLConditions, #include-expanded). z0-specific
                    // keys parse always; the generic-named ones (d/kappa/Cmu) only when this is an ABL entry.
                    else if (key == "Uref")
                    {
                        p.ablUref = ts.nextScalar();
                        ts.expect(";");
                    }
                    else if (key == "Zref")
                    {
                        p.ablZref = ts.nextScalar();
                        ts.expect(";");
                    }
                    else if (key == "z0")
                    {
                        if (ts.peek() == "uniform" || ts.peek() == "constant") ts.next();
                        p.ablZ0 = ts.nextScalar();
                        ts.expect(";");
                    }
                    else if (key == "flowDir" || key == "zDir")
                    {
                        ts.expect("(");
                        const vector v{ts.nextScalar(), ts.nextScalar(), ts.nextScalar()};
                        ts.expect(")");
                        ts.expect(";");
                        if (key == "flowDir") p.ablFlowDir = v;
                        else p.ablZDir = v;
                    }
                    else if (key == "d" && p.hasABL)
                    {
                        if (ts.peek() == "uniform" || ts.peek() == "constant") ts.next();
                        p.ablD = ts.nextScalar();
                        ts.expect(";");
                    }
                    else if ((key == "kappa" || key == "Cmu") && p.hasABL)
                    {
                        const scalar v = ts.nextScalar();
                        ts.expect(";");
                        if (key == "kappa") p.ablKappa = v;
                        else p.ablCmu = v;
                    }
                    // surfaceNormalFixedValue refValue / uniformNormalFixedValue uniformValue: SCALAR (U_b = refValue * n).
                    else if ((key == "refValue" && p.type == "surfaceNormalFixedValue") ||
                             (key == "uniformValue" && p.type == "uniformNormalFixedValue"))
                    {
                        const std::string m = ts.next();     // uniform <s> | constant <s> | nonuniform List<scalar> | table(...)
                        if (m == "uniform" || m == "constant")
                        {
                            p.normalRefUniform = true;
                            p.normalRefUniformValue = ts.nextScalar();
                            p.hasNormalRef = true;
                        }
                        else if (m == "nonuniform")
                        {
                            p.normalRefUniform = false;
                            ts.next();
                            const label n = ts.nextLabel();
                            ts.expect("(");
                            p.normalRefValues.resize(n);
                            for (label i = 0; i < n; ++i)
                                p.normalRefValues[i] = ts.nextScalar();
                            ts.expect(")");
                            p.hasNormalRef = true;
                        }
                        else   // table/Function1 -> ramp handles it; treat as 0
                        {
                            skipToSemicolon(ts, m == "(" ? 1 : 0);
                        }
                        ts.expect(";");
                    }
                    else if (key == "inletValue")   // inletOutlet inflow value (may be $internalField)
                    {
                        readValueOrInternal(ts, fd, p.inletUniform, p.inletUniformValue, p.inletValues);
                        p.hasInletValue = true;
                        ts.expect(";");
                    }
                    else if (key == "p0")   // totalPressure reference p0 (reuse the inletValue slot)
                    {
                        readValueOrInternal(ts, fd, p.inletUniform, p.inletUniformValue, p.inletValues);
                        p.hasInletValue = true;
                        ts.expect(";");
                    }
                    else if (key == "uniformValue")   // uniformFixedValue: steady PatchFunction1 "constant <v>"
                    {
                        const std::string m = ts.next();     // "constant" | "uniform" (steady forms; tables unsupported)
                        if (m == "constant" || m == "uniform")
                        {
                            p.uniformValue = readFoamValue<T>(ts);
                            p.valueUniform = true;
                            p.hasValue = true;
                        }
                        else   // table/coded/Function1 -> skip (paren-aware); dispatch will throw if no value
                        {
                            skipToSemicolon(ts, m == "(" ? 1 : 0);
                        }
                        ts.expect(";");
                    }
                    else if (key == "value")
                    {
                        readValueOrInternal(ts, fd, p.valueUniform, p.uniformValue, p.values);   // uniform/nonuniform/$internalField
                        p.hasValue = true;
                        ts.expect(";");
                    }
                    else if (key == "intensity")   // turbulentIntensityKineticEnergyInlet
                    {
                        p.intensity = ts.nextScalar();
                        ts.expect(";");
                    }
                    else if (key == "mixingLength")   // turbulentMixingLength*Inlet
                    {
                        p.mixingLength = ts.nextScalar();
                        ts.expect(";");
                    }
                    else if (key == "freestreamValue")   // freestream/freestreamPressure farfield value (may be $internalField)
                    {
                        readValueOrInternal(ts, fd, p.valueUniform, p.uniformValue, p.values);
                        p.hasValue = true;
                        p.inletUniform = p.valueUniform;
                        p.inletUniformValue = p.uniformValue;
                        p.inletValues = p.values;
                        p.hasInletValue = true;
                        ts.expect(";");
                    }
                    else
                    {
                        skipToSemicolon(ts);                 // skip any other (unhandled) entry up to its ';' (paren-aware)
                        ts.expect(";");
                    }
                }
                ts.expect("}");
                if (p.type == "timeVaryingMappedFixedValue")
                {
                    // OF requires constant/boundaryData/<patch>; do NOT swallow a read failure into a silent
                    // zeroGradient/fixedValue fallback -- rethrow with context so the misconfiguration is detected.
                    try
                    {
                        readTimeVaryingMapped(path, p.name, p);
                    }
                    catch (const std::exception& e)
                    {
                        throw std::runtime_error("brae: timeVaryingMappedFixedValue on patch '" + p.name +
                            "' requires constant/boundaryData/" + p.name + " (points + a time dir); read failed: " + e.what());
                    }
                }
                fd.boundary.push_back(std::move(p));
            }
            ts.expect("}");
        }
        // dimensions and other top-level entries are skipped token-by-token.
    }
    return fd;
}

} // namespace brae
