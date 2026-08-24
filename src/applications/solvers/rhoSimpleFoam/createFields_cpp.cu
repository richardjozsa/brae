// _cpp REFERENCE implementation -- see createFields_cpp.cuh for the OpenFOAM provenance.
#include "createFields_cpp.cuh"
#include "foam_field_reader.cuh"
#include "thermo_parse.cuh"
#include "equation_of_state.cuh"
#include <algorithm>
#include <filesystem>
#include <limits>
#include <stdexcept>

namespace brae {
namespace cpu {
// rhoSimpleFoam's own components. Namespaced because simpleFoam has an assembleUEqn, a createFields and
// a pEqn of its own: the two solvers transcribe DIFFERENT OpenFOAM files that happen to share names, and
// letting them collide in one namespace would make which one a call site gets an accident of includes.
namespace rhoSimple {

namespace {

const scalar kGreat = 1.0e15;   // OpenFOAM GREAT, as pressureControl.C initialises pMax_/pMin_ with

bool fileExists(const std::string& path)
{
    return std::filesystem::exists(path) || std::filesystem::exists(path + ".gz");
}

// GeometricField.C:1073-1084 -- true unless some patch fixes a value.
bool needReference(const GeometricField<scalar>& p)
{
    for (const auto& b : p.boundary)
        if (b->fixesValue()) return false;
    return true;
}

// findRefCell.C:33-119. Returns true when a reference WAS set, which is what pressureControl uses to
// decide whether a reference pressure is available for the *Factor limits.
bool setRefCell(
    const GeometricField<scalar>& p,
    const FoamDict*               dict,
    label                         nCells,
    label&                        refCell,
    scalar&                       refValue)
{
    if (!needReference(p)) return false;
    if (!dict)
        throw std::runtime_error(
            "rhoSimpleFoam createFields: p needs a reference (no boundary patch fixes its value) but "
            "fvSolution has no SIMPLE dictionary to read pRefCell/pRefValue from (findRefCell.C:33).");

    if (dict->found("pRefCell"))
    {
        refCell = dict->intOr("pRefCell", 0);
        if (refCell < 0 || refCell >= nCells)
            throw std::runtime_error(
                "rhoSimpleFoam createFields: illegal pRefCell " + std::to_string(refCell)
                + "; should be 0.." + std::to_string(nCells) + " (findRefCell.C:55-62).");
    }
    else if (dict->found("pRefPoint"))
    {
        // OpenFOAM locates the cell containing pRefPoint (mesh.findCell). brae has no point-location
        // search on this path, and guessing a cell would silently pin the pressure level somewhere the
        // user did not ask for.
        throw std::runtime_error(
            "rhoSimpleFoam createFields: pRefPoint is set. OpenFOAM resolves it with mesh.findCell "
            "(findRefCell.C:69-100); brae has no cell search here. Use pRefCell instead.");
    }
    else
    {
        throw std::runtime_error(
            "rhoSimpleFoam createFields: p needs a reference (no boundary patch fixes its value) but "
            "neither pRefCell nor pRefPoint is set in the SIMPLE dictionary (findRefCell.C).");
    }
    refValue = dict->scalarOr("pRefValue", 0.0);
    return true;
}

// pressureControl.C:33-190, in OpenFOAM's own order.
PressureControl makePressureControl(
    const GeometricField<scalar>& p,
    const GeometricField<scalar>& rho,
    const FoamDict*               dict,
    label                         nCells)
{
    PressureControl pc;
    pc.pMax = kGreat;
    pc.pMin = 0.0;

    bool   pLimits = false;
    scalar pMax = -kGreat;
    scalar pMin = kGreat;

    if (setRefCell(p, dict, nCells, pc.refCell, pc.refValue))
    {
        pLimits = true;
        pMax = pc.refValue;
        pMin = pc.refValue;
    }
    if (!dict) return pc;

    // pMax AND pMin together short-circuit everything below -- no boundary scan, no factors.
    if (dict->found("pMax") && dict->found("pMin"))
    {
        pc.pMax = dict->scalarOr("pMax", kGreat);
        pc.limitMaxP = true;
        pc.pMin = dict->scalarOr("pMin", 0.0);
        pc.limitMinP = true;
        return pc;
    }

    // Otherwise the reference pressure (and density) come from the patches that FIX a value.
    scalar rhoRefMax = -kGreat;
    scalar rhoRefMin = kGreat;
    bool   rhoLimits = false;
    for (std::size_t pi = 0; pi < p.boundary.size(); ++pi)
    {
        if (!p.boundary[pi]->fixesValue()) continue;
        const std::vector<scalar>& pv = p.boundary[pi]->value();
        if (pv.empty()) continue;
        pLimits   = true;
        rhoLimits = true;
        pMax = std::max(pMax, *std::max_element(pv.begin(), pv.end()));
        pMin = std::min(pMin, *std::min_element(pv.begin(), pv.end()));
        if (pi < rho.boundary.size())
        {
            const std::vector<scalar>& rv = rho.boundary[pi]->value();
            if (!rv.empty())
            {
                rhoRefMax = std::max(rhoRefMax, *std::max_element(rv.begin(), rv.end()));
                rhoRefMin = std::min(rhoRefMin, *std::min_element(rv.begin(), rv.end()));
            }
        }
    }

    // The MAXIMUM: pMax, else pMaxFactor * reference, else the backward-compatible rhoMax.
    if (dict->found("pMax"))
    {
        pc.pMax = dict->scalarOr("pMax", kGreat);
        pc.limitMaxP = true;
    }
    else if (dict->found("pMaxFactor"))
    {
        if (!pLimits)
            throw std::runtime_error(
                "rhoSimpleFoam createFields: 'pMaxFactor' specified rather than 'pMax', but the "
                "corresponding reference pressure cannot be evaluated from the boundary conditions "
                "(no patch fixes p and no pRefCell). Specify 'pMax' rather than 'pMaxFactor' "
                "(pressureControl.C:96-105).");
        pc.pMax = pMax * dict->scalarOr("pMaxFactor", 1.0);
        pc.limitMaxP = true;
    }
    else if (dict->found("rhoMax"))
    {
        // OpenFOAM warns and keeps going; brae keeps the same behaviour but says it out loud, because
        // the limit that results is scaled off a boundary density rather than one the user wrote.
        if (!pLimits)
            throw std::runtime_error(
                "rhoSimpleFoam createFields: 'rhoMax' specified rather than 'pMax', but the "
                "corresponding reference pressure cannot be evaluated from the boundary conditions "
                "(pressureControl.C:112-126).");
        if (!rhoLimits)
            throw std::runtime_error(
                "rhoSimpleFoam createFields: 'rhoMax' specified rather than 'pMaxFactor', but the "
                "corresponding reference density cannot be evaluated from the boundary conditions "
                "(pressureControl.C:127-137).");
        const scalar rhoMax = dict->scalarOr("rhoMax", kGreat);
        pc.pMax = std::max(rhoMax / rhoRefMax, (scalar)1.0) * pMax;
        pc.limitMaxP = true;
    }

    // The MINIMUM: the same three, mirrored.
    if (dict->found("pMin"))
    {
        pc.pMin = dict->scalarOr("pMin", 0.0);
        pc.limitMinP = true;
    }
    else if (dict->found("pMinFactor"))
    {
        if (!pLimits)
            throw std::runtime_error(
                "rhoSimpleFoam createFields: 'pMinFactor' specified rather than 'pMin', but the "
                "corresponding reference pressure cannot be evaluated from the boundary conditions "
                "(pressureControl.C:145-155).");
        pc.pMin = pMin * dict->scalarOr("pMinFactor", 1.0);
        pc.limitMinP = true;
    }
    else if (dict->found("rhoMin"))
    {
        if (!pLimits)
            throw std::runtime_error(
                "rhoSimpleFoam createFields: 'rhoMin' specified rather than 'pMin', but the "
                "corresponding reference pressure cannot be evaluated from the boundary conditions "
                "(pressureControl.C:162-176).");
        if (!rhoLimits)
            throw std::runtime_error(
                "rhoSimpleFoam createFields: 'rhoMin' specified rather than 'pMinFactor', but the "
                "corresponding reference density cannot be evaluated from the boundary conditions "
                "(pressureControl.C:177-187).");
        const scalar rhoMin = dict->scalarOr("rhoMin", 0.0);
        pc.pMin = std::min(rhoMin / rhoRefMin, (scalar)1.0) * pMin;
        pc.limitMinP = true;
    }
    return pc;
}

}   // namespace


bool PressureControl::limit(std::vector<scalar>& p) const
{
    if (!limitMaxP && !limitMinP) return false;
    if (limitMaxP)
    {
        for (scalar& v : p) v = std::min(v, pMax);
    }
    if (limitMinP)
    {
        for (scalar& v : p) v = std::max(v, pMin);
    }
    // OpenFOAM returns true whenever a limit is ACTIVE, not whether a value actually moved -- the caller
    // uses it to decide whether to re-evaluate p's boundary conditions.
    return true;
}


RhoSimpleFields createFields(
    const std::string&          timeDir,
    const std::string&          caseDir,
    const FoamDict*             simpleDict,
    const FoamDict*             fvSolution,
    const PrimitiveMesh&        m,
    const FvGeometry&           g,
    const std::vector<FvPatch>& patches)
{
    RhoSimpleFields f;
    const label nC = m.nCells();

    // fluidThermo::New(mesh). readThermoCoeffs refuses an unsupported thermo BY NAME rather than falling
    // back to a default, so an unhandled equation of state stops here instead of silently running as a
    // perfect gas.
    f.thermo = readThermoCoeffs(caseDir, fvSolution);

    // thermo.validate(args.executable(), "h", "e") -- rhoSimpleFoam accepts exactly these two energy
    // variables, because EEqn.H's kinetic-energy source is written for both and for nothing else.
    {
        const FoamDict tp = readDict(caseDir + "/constant/thermophysicalProperties");
        const FoamDict* tt = tp.subDict("thermoType");
        const std::string energy = tt ? tt->wordOr("energy", "") : "";
        if (energy == "sensibleEnthalpy")            f.heName = "h";
        else if (energy == "sensibleInternalEnergy") f.heName = "e";
        else
            throw std::runtime_error(
                "brae: rhoSimpleFoam thermo energy '" + (energy.empty() ? std::string("<missing>") : energy)
                + "' is not one this solver transports (sensibleEnthalpy -> h, sensibleInternalEnergy -> e)."
                  " OpenFOAM's thermo.validate(.., \"h\", \"e\") refuses the same set. Refusing rather than"
                  " solving a different energy equation.");
    }

    // p = thermo.p() and T: both read by the thermo, both MUST_READ. Read together so psi and rho below
    // are derived from the SAME state rather than from two fields that could be an iteration apart.
    f.p = buildField<scalar>(readField<scalar>(timeDir + "/p"), patches, nC);
    f.p.evaluateBoundary();
    f.T = buildField<scalar>(readField<scalar>(timeDir + "/T"), patches, nC);
    f.T.evaluateBoundary();

    // rho: READ_IF_PRESENT, else thermo.rho(). A restart continues from the written density; a cold start
    // computes it from the equation of state.
    const std::string rhoPath = timeDir + "/rho";
    f.rhoWasRead = fileExists(rhoPath);
    if (f.rhoWasRead)
    {
        f.rho = buildField<scalar>(readField<scalar>(rhoPath), patches, nC);
        f.rho.evaluateBoundary();
    }
    else
    {
        // thermo.rho() with the boundary values taken from the boundary p and T, so rho's patch values are
        // the equation of state's and not a copy of the internal cell's.
        FieldData<scalar> fd;
        fd.internalUniform = false;   // defaults TRUE; a hand-built field must say so
        fd.internalField.resize(nC);
        for (label c = 0; c < nC; ++c)
            fd.internalField[c] = perfectGasRho(f.p.internal[c], f.T.internal[c], f.thermo);
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            PatchFieldData<scalar> b;
            b.name     = patches[pi].name;
            b.type     = "calculated";
            b.hasValue = true;
            const std::vector<scalar>& pb = f.p.boundary[pi]->value();
            const std::vector<scalar>& tb = f.T.boundary[pi]->value();
            b.values.resize(patches[pi].size);
            for (label i = 0; i < patches[pi].size; ++i)
                b.values[i] = perfectGasRho(pb[i], tb[i], f.thermo);
            fd.boundary.push_back(std::move(b));
        }
        f.rho = buildField<scalar>(fd, patches, nC);
    }

    // U: MUST_READ.
    f.U = buildField<vector>(readField<vector>(timeDir + "/U"), patches, nC);
    f.U.evaluateBoundary();

    // compressibleCreatePhi.H: READ_IF_PRESENT, else linearInterpolate(rho*U) & Sf.
    //
    // The PRODUCT is interpolated, not the two factors separately -- see the header. rho*U is built per
    // cell and per boundary face and fluxed through the ordinary linear-interpolation path, so the face
    // weights are the ones fvc::flux already reproduces from OpenFOAM.
    const std::string phiPath = timeDir + "/phi";
    f.phiWasRead = fileExists(phiPath);
    if (f.phiWasRead)
    {
        const FieldData<scalar> pf = readField<scalar>(phiPath);
        f.phi.internal = pf.internalField;
        f.phi.boundary.resize(patches.size());
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            f.phi.boundary[pi].assign(patches[pi].size, 0.0);
            for (const auto& b : pf.boundary)
                if (b.name == patches[pi].name && b.hasValue
                    && static_cast<label>(b.values.size()) == patches[pi].size)
                    f.phi.boundary[pi] = b.values;
        }
    }
    else
    {
        std::vector<vector> rhoU(nC);
        for (label c = 0; c < nC; ++c)
        {
            const scalar r = f.rho.internal[c];
            rhoU[c] = vector{ r * f.U.internal[c].x, r * f.U.internal[c].y, r * f.U.internal[c].z };
        }
        std::vector<std::vector<vector>> rhoUb(patches.size());
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            const std::vector<scalar>& rb = f.rho.boundary[pi]->value();
            const std::vector<vector>& ub = f.U.boundary[pi]->value();
            rhoUb[pi].resize(patches[pi].size);
            for (label i = 0; i < patches[pi].size; ++i)
                rhoUb[pi][i] = vector{ rb[i] * ub[i].x, rb[i] * ub[i].y, rb[i] * ub[i].z };
        }
        f.phi = fvc::flux(rhoU, rhoUb, m, g, patches);
    }

    // createFieldRefs.H: psi is thermo.psi(). Derived from the same T that rho came from.
    f.psi.resize(nC);
    for (label c = 0; c < nC; ++c) f.psi[c] = perfectGasPsi(f.T.internal[c], f.thermo);
    f.psiBnd.resize(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const std::vector<scalar>& tb = f.T.boundary[pi]->value();
        f.psiBnd[pi].resize(patches[pi].size);
        for (label i = 0; i < patches[pi].size; ++i)
            f.psiBnd[pi][i] = perfectGasPsi(tb[i], f.thermo);
    }

    // thermo.he() -- the variable EEqn transports. hConstThermo: Hs = Cp*(T - Tref) + Href, and the
    // heat of formation Hf belongs to the ABSOLUTE enthalpy only, so it must not appear here.
    f.he.resize(nC);
    for (label c = 0; c < nC; ++c)
    {
        const scalar hs = f.thermo.Cp * (f.T.internal[c] - f.thermo.Tref) + f.thermo.Href;
        // e = h - p/rho = h - R*T for a perfect gas.
        f.he[c] = (f.heName == "e") ? hs - f.thermo.R * f.T.internal[c] : hs;
    }

    f.pressureControl = makePressureControl(f.p, f.rho, simpleDict, nC);

    // initialMass = fvc::domainIntegrate(rho). pEqn.H's closed-volume correction is measured against it.
    f.initialMass = 0.0;
    for (label c = 0; c < nC; ++c) f.initialMass += f.rho.internal[c] * g.V()[c];

    return f;
}

} // namespace rhoSimple
} // namespace cpu
} // namespace brae
