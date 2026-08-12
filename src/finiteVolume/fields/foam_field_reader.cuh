#pragma once
// Reads an OpenFOAM field file: dimensions / internalField (uniform|nonuniform) /
// boundaryField { patch { type; value; ... } }. Templated on the value type (scalar/vector).
// ASCII for now; binary field values land with the binary field reader (later increment).
#include "cf_types.cuh"
#include "function1.cuh"   // OF Function1: constant / table
#include "foam_token_reader.cuh"
#include <string>
#include <type_traits>
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
    // uniformFixedValue whose uniformValue is a Function1 (table / polynomial / coded / expression)
    // rather than a constant. brae cannot evaluate those, and the danger is that the entry ALSO carries a
    // stale `value` from an overridden fixedValue entry, so the field silently takes that constant
    // instead. Recorded here and refused at construction rather than guessed at.
    std::string    unsupportedFunction1;
    Function1      p0Function1;              // uniformTotalPressure p0(t); empty unless hasP0Function1
    bool           hasP0Function1 = false;
    // pressureInletOutletVelocity's optional `tangentialVelocity`. OF sets
    // refValue = tv - n*(n & tv) (pressureInletOutletVelocityFvPatchVectorField.C:135); brae leaves the
    // tangential refValue at zero, so honouring the key would need per-face storage it does not have.
    // Recorded so it can be refused instead of quietly changing the boundary condition.
    bool           hasTangentialVelocity = false;
    // fixedGradient (and the heat-flux BCs derived from it): the prescribed normal gradient.
    // Plain `mixed` (Robin) carries refValue + refGradient + valueFraction. refGradient shares the
    // gradient* slots below; these two are its own. Distinct from inletValue*, which inletOutlet and the
    // freestream family use for a refValue whose blend the DEVICE recomputes each step.
    bool           hasRefValue      = false;
    bool           refValueUniform  = false;
    T              refValueUniformValue{};
    std::vector<T> refValues;
    bool           hasValueFraction = false;
    bool           vfUniform        = false;
    scalar         vfUniformValue   = 0;
    std::vector<scalar> vfValues;

    bool           hasGradient    = false;
    bool           gradientUniform = false;
    T              gradientUniformValue{};
    std::vector<T> gradientValues;
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
    // compressible::alphatWallFunction: alphat_w = rho_w*nut_w/Prt. OF's default here is 0.85
    // (alphatWallFunctionFvPatchScalarField.C: dict.getOrDefault<scalar>("Prt", 0.85)) and is NOT the
    // turbulence model's own Prt, whose default is 1.0. Two different numbers in the same case.
    scalar         Prt          = 0.85;
    // totalPressure: OF picks its formula from these. psi "none" (the default) -> the low-speed form
    // p0 - 0.5*rho*neg(phi)*|U|^2; a NAMED psi -> the isentropic high-speed form with gamma, which brae
    // does not implement. Recorded so it can be refused by name instead of silently running low-speed.
    std::string    psiName      = "none";
    scalar         gammaTP      = 1.0;
    // flowRateInletVelocity (OF flowRateInletVelocityFvPatchVectorField). OF selects the branch by which
    // key is present: "volumetricFlowRate" -> volumetric_ = true; otherwise "massFlowRate" (default
    // rhoName "rho"). rhoInlet is only the FALLBACK used when the rho field is not registered -- in
    // rhoSimpleFoam it is, so the real patch rho is used and rhoInlet is ignored, exactly as OF does.
    bool           hasFlowRate  = false;
    bool           flowRateIsMass = false;
    scalar         flowRate     = 0.0;
    scalar         rhoInlet     = -1.0;   // OF default -VGREAT ("not given")
    bool           extrapolateProfile = false;
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
    bool   atmBoundNut = true;   // atmNutkWallFunction boundNut option (clamp nut>=0); z0 is stored in ablZ0.
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
// Does this token start a numeric literal? Used to spot a bare (keyword-less) value entry.
inline bool isFoamNumber(const std::string& t)
{
    if (t.empty()) return false;
    const char c = t[0];
    return (c >= '0' && c <= '9') || c == '-' || c == '+' || c == '.';
}
// OF Function1 accepts a BARE value as shorthand for `constant <v>`: `uniformValue (0 0 0);` and
// `uniformValue 5;` are constants, not tables. brae required the keyword, so a bare vector was
// classified as an unsupported Function1 and the case refused -- simpleFoam/turbineSiting's terrain
// patch is exactly `uniformFixedValue` with `uniformValue (0 0 0)`.
template <typename T>
inline T readBareFoamValue(TokenStream& ts, const std::string& first);

template <>
inline scalar readBareFoamValue<scalar>(TokenStream& ts, const std::string& first)
{
    (void)ts;
    return static_cast<scalar>(std::strtod(first.c_str(), nullptr));
}

template <>
inline vector readBareFoamValue<vector>(TokenStream& ts, const std::string& first)
{
    // `first` is already the '('; the three components and the ')' remain.
    (void)first;
    vector v{};
    v.x = ts.nextScalar();
    v.y = ts.nextScalar();
    v.z = ts.nextScalar();
    ts.expect(")");
    return v;
}


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
    // A BARE value, i.e. no `uniform`/`nonuniform` keyword: `inletValue (0 0 0);`. OpenFOAM tolerates
    // these because a boundary condition only reads the entries it cares about -- gasMixing's
    // pressureInletOutletVelocity carries an `inletValue` that OF's implementation never looks at, so OF
    // runs the case fine. brae parsed every known key unconditionally and died with
    // "TokenStream: expected '(' got '0'", failing on a file OpenFOAM accepts and, worse, failing BEFORE
    // reaching its own legitimate refusal (nutUWallFunction), so the message pointed at the wrong thing.
    else if (ts.peek() == "(" || isFoamNumber(ts.peek()))
    {
        uniform = true;
        uval = readFoamValue<T>(ts);
        vals.clear();
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
                const std::string pname = ts.next();
                if (ts.peek() != "{")
                {
                    // Non-sub-dict entry at the boundaryField level: a leftover variable definition spliced from an
                    // #include'd ABLConditions/initialConditions at that scope (e.g. "Uref 10.0;", referenced later
                    // via $z0 -- turbineSiting). OpenFOAM keeps these as dict variables; brae's $-expansion has already
                    // resolved the references, so skip the definition to its ';'. Real patches are always "name { ... }".
                    skipToSemicolon(ts);
                    ts.expect(";");
                    continue;
                }
                PatchFieldData<T> p;
                p.name = pname;
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
                    else if (key == "boundNut")   // atmNutkWallFunction: clamp nut>=0 (true/false)
                    {
                        const std::string v = ts.next();
                        p.atmBoundNut = (v == "true" || v == "yes" || v == "on" || v == "1");
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
                        // p0 is a Function1 on uniformTotalPressure: it may be `table (...)`,
                        // `polynomial`, `csvFile`, an expression -- not just a field value.
                        // readValueOrInternal only knows uniform/nonuniform, so a table made the
                        // TOKENISER fail mid-parse ("expected '(' got '0'", pimpleFoam/RAS/TJunction).
                        // That is a raw parser error where this codebase's rule is that unsupported
                        // input is NAMED. Record it and let the dispatch refuse by name instead.
                        const std::string m = ts.peek();
                        if (m == "uniform" || m == "nonuniform")
                        {
                            readValueOrInternal(ts, fd, p.inletUniform, p.inletUniformValue, p.inletValues);
                            p.hasInletValue = true;
                        }
                        else if (m == "constant")
                        {
                            ts.next();
                            p.inletUniformValue = readFoamValue<T>(ts);
                            p.inletUniform = true;
                            p.hasInletValue = true;
                        }
                        else if (isFoamNumber(m))
                        {
                            p.inletUniformValue = readBareFoamValue<T>(ts, ts.next());
                            p.inletUniform = true;
                            p.hasInletValue = true;
                        }
                        else if (m == "table")
                        {
                            // OF Function1 `table ((t v) (t v) ...)`: linear between entries, CLAMPed
                            // outside (TableBase.C:76). pimpleFoam/RAS/TJunction ramps p0 this way.
                            ts.next();                       // "table"
                            ts.expect("(");
                            std::vector<std::pair<scalar, scalar>> pts;
                            while (!ts.eof() && ts.peek() != ")")
                            {
                                ts.expect("(");
                                const scalar tt = ts.nextScalar();
                                const scalar vv = ts.nextScalar();
                                ts.expect(")");
                                pts.emplace_back(tt, vv);
                            }
                            ts.expect(")");
                            p.p0Function1 = Function1::table(std::move(pts));
                            p.hasP0Function1 = true;
                            // NOT YET WIRED: DeviceSimpleSolver::setTimeVaryingP0 exists and the
                            // per-step device refresh is in place, but no driver hands the tables over,
                            // so p0 would stay frozen at its t=0 value while the case believes it is
                            // ramping. Measured on pimpleFoam/RAS/TJunction: inlet p fell 9.32 -> 8.62
                            // where the table asks for p0 13.09 -> 15.11. A plausible wrong answer is
                            // worse than a refusal, so keep naming it until the driver wiring lands.
                            p.unsupportedFunction1 = "table (parsed, but p0(t) is not yet applied)";
                            // Seed the constant slot with t = 0 so a solver that never advances time
                            // still has a defined p0 rather than zero.
                            // p0 is a PRESSURE: scalar only. The reader is templated on T, so guard
                            // rather than cast -- a vector field has no p0 and must not silently get one.
                            if constexpr (std::is_same_v<T, scalar>)
                            {
                                p.inletUniformValue = p.p0Function1.value(0);
                                p.inletUniform = true;
                                p.hasInletValue = true;
                            }
                        }
                        else
                        {
                            p.unsupportedFunction1 = m;
                            ts.next();
                            skipToSemicolon(ts, 0);
                        }
                        ts.expect(";");
                    }
                    else if (key == "uniformValue")   // uniformFixedValue: steady PatchFunction1 "constant <v>"
                    {
                        const std::string m = ts.next();     // "constant" | "uniform" | a BARE value
                        if (m == "constant" || m == "uniform")
                        {
                            p.uniformValue = readFoamValue<T>(ts);
                            p.valueUniform = true;
                            p.hasValue = true;
                        }
                        else if (m == "(" || isFoamNumber(m))
                        {
                            // Bare constant (see readBareFoamValue): OF's Function1 shorthand.
                            p.uniformValue = readBareFoamValue<T>(ts, m);
                            p.valueUniform = true;
                            p.hasValue = true;
                        }
                        else   // table / polynomial / coded / expression: skip the entry, then REFUSE.
                        {
                            // Relying on "dispatch throws when there is no value" is not enough: a case
                            // that overrides an earlier `type fixedValue; value uniform X;` still has
                            // hasValue == true, so the Function1 silently degrades to the constant X.
                            // squareBendLiq does exactly that (T walls: expression, stale value 350).
                            // A dict form ({ type expression; ... }) names its Function1 inside, so peek
                            // the `type` keyword -- "a dictionary" is a much worse error message than
                            // "expression".
                            p.unsupportedFunction1 = m;
                            if (m == "{")
                            {
                                const std::string t1 = ts.peek();
                                if (t1 == "type")
                                {
                                    ts.next();
                                    p.unsupportedFunction1 = ts.peek();
                                }
                            }
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
                    // OF takes a Function1 here; "constant <v>" and a bare "<v>" are the steady forms.
                    // Anything else (table/polynomial/...) is refused by name rather than approximated.
                    else if (key == "volumetricFlowRate" || key == "massFlowRate")
                    {
                        p.hasFlowRate = true;
                        p.flowRateIsMass = (key == "massFlowRate");
                        std::string w = ts.next();
                        if (w == "constant") w = ts.next();
                        else if (w == "{")
                            throw std::runtime_error(
                                "brae: flowRateInletVelocity '" + key + "' given as a Function1 dictionary on patch "
                                + p.name + "; only 'constant <value>' (or a bare value) is supported.");
                        p.flowRate = std::stod(w);
                        ts.expect(";");
                    }
                    else if (key == "rhoInlet")
                    {
                        p.rhoInlet = ts.nextScalar();
                        ts.expect(";");
                    }
                    else if (key == "extrapolateProfile")
                    {
                        const std::string w = ts.next();
                        p.extrapolateProfile = (w == "true" || w == "yes" || w == "on" || w == "1");
                        ts.expect(";");
                    }
                    else if (key == "psi")    // totalPressure: selects OF's isentropic branch when != none
                    {
                        p.psiName = ts.next();
                        ts.expect(";");
                    }
                    else if (key == "gamma")   // totalPressure isentropic exponent
                    {
                        p.gammaTP = ts.nextScalar();
                        ts.expect(";");
                    }
                    else if (key == "Prt")   // compressible::alphatWallFunction turbulent Prandtl number
                    {
                        p.Prt = ts.nextScalar();
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
                    // `gradient` is fixedGradient's key, `refGradient` is mixed's. They fill the same slot
                    // because they mean the same thing to the discretisation -- the difference is the (1-vf)
                    // weight the kernels apply on a mixed patch (OF mixedFvPatchField.C:279-310).
                    else if (key == "gradient" || key == "refGradient")
                    {
                        readValueOrInternal(ts, fd, p.gradientUniform, p.gradientUniformValue, p.gradientValues);
                        p.hasGradient = true;
                        ts.expect(";");
                    }
                    else if (key == "refValue" && p.type == "mixed")
                    {
                        readValueOrInternal(ts, fd, p.refValueUniform, p.refValueUniformValue, p.refValues);
                        p.hasRefValue = true;
                        ts.expect(";");
                    }
                    else if (key == "valueFraction")
                    {
                        // Always a SCALAR field, even on a vector patch (OF blends component-wise with one
                        // fraction), so it cannot go through readValueOrInternal<T>.
                        const std::string w = ts.peek();
                        if (w == "uniform")
                        {
                            ts.next();
                            p.vfUniform = true;
                            p.vfUniformValue = std::stod(ts.next());
                        }
                        else
                        {
                            ts.next();                      // nonuniform
                            if (ts.peek() == "List<scalar>") ts.next();
                            const int n = std::stoi(ts.next());
                            ts.expect("(");
                            p.vfValues.resize(static_cast<std::size_t>(n));
                            for (int i = 0; i < n; ++i) p.vfValues[static_cast<std::size_t>(i)] = std::stod(ts.next());
                            ts.expect(")");
                        }
                        p.hasValueFraction = true;
                        ts.expect(";");
                    }
                    else if (key == "tangentialVelocity")   // pressureInletOutletVelocity, optional
                    {
                        p.hasTangentialVelocity = true;
                        skipToSemicolon(ts);
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
