// Direct, no-solve comparison of Brae's kOmegaSST + nutkWallFunction yPlus calculation with a retained
// OpenFOAM field. The executable accepts freshly extracted case directories; it never writes the cases.
#include "brae_yplus.cuh"
#include "device_simple_foam.cuh"
#include "near_wall_dist.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using namespace brae;

namespace {

// Declared before any retained values are inspected. The retained files use writePrecision 8; this allows one
// half-unit in the last printed decimal plus propagation of the same precision through U/k/nut, while remaining
// tight enough to reject a different wall-function formula. Relative error excludes values at/under 1e-12.
constexpr scalar kNearZero = 1e-12;
constexpr scalar kAbsTol = 1e-5;
constexpr scalar kRelTol = 5e-7;

template <typename T>
const PatchFieldData<T>* exactBoundary(const FieldData<T>& f, const std::string& name)
{
    for (const auto& b : f.boundary) if (b.name == name) return &b;
    return nullptr;
}

template <typename T>
std::vector<T> internalValues(const FieldData<T>& f, label n)
{
    if (f.internalUniform) return std::vector<T>(static_cast<std::size_t>(n), f.internalUniformValue);
    if (static_cast<label>(f.internalField.size()) != n)
        throw std::runtime_error("retained field internal size does not match mesh");
    return f.internalField;
}

template <typename T>
std::vector<T> boundaryValues(const FieldData<T>& f, const std::string& name, label n)
{
    const auto* b = exactBoundary(f, name);
    if (!b) throw std::runtime_error("retained field has no boundary entry for patch '" + name + "'");
    if (!b->hasValue && b->type == "noSlip")
        return std::vector<T>(static_cast<std::size_t>(n), T{});
    if (b->valueUniform) return std::vector<T>(static_cast<std::size_t>(n), b->uniformValue);
    if (static_cast<label>(b->values.size()) != n)
        throw std::runtime_error("retained field boundary value count does not match patch '" + name + "'");
    return b->values;
}

scalar areaMean(const std::vector<scalar>& v, const FvPatch& p)
{
    scalar area = 0, sum = 0;
    for (label i = 0; i < p.size; ++i) { area += p.magSf[i]; sum += p.magSf[i]*v[static_cast<std::size_t>(i)]; }
    return sum/area;
}

scalar bandPercent(const std::vector<scalar>& v, const FvPatch& p)
{
    scalar area = 0, band = 0;
    for (label i = 0; i < p.size; ++i)
    {
        area += p.magSf[i];
        if (v[static_cast<std::size_t>(i)] >= 30 && v[static_cast<std::size_t>(i)] <= 300) band += p.magSf[i];
    }
    return 100*band/area;
}

bool requiredPatch(const std::string& name)
{
    return name == "body" || name == "lowerWall";
}

int compareCase(const std::string& caseDir, const std::string& runLabel)
{
    namespace fs = std::filesystem;
    const fs::path marker = fs::path(caseDir) / ".fresh-extraction-marker";
    if (!fs::is_regular_file(marker))
        throw std::runtime_error(runLabel + ": missing fresh-extraction marker; refusing cross-case evidence");
    for (const std::string& field : {"U", "k", "nut", "yPlus"})
    {
        const fs::path plain = fs::path(caseDir) / "2000" / field;
        const fs::path gz = fs::path(caseDir) / "2000" / (field + ".gz");
        if (!fs::exists(plain) && !fs::exists(gz))
            throw std::runtime_error(runLabel + ": missing same-case 2000/" + field);
    }

    const FoamDict transport = readDict(caseDir + "/constant/transportProperties");
    const FoamDict turb = readDict(caseDir + "/constant/turbulenceProperties");
    const FoamDict* ras = turb.subDict("RAS");
    const std::string model = ras ? ras->wordOr("model", ras->wordOr("RASModel", "")) : "";
    if (model != "kOmegaSST")
        throw std::runtime_error(runLabel + ": retained turbulence model is '" + model + "', not kOmegaSST");
    const scalar nu = transport.scalarOr("nu", -1);
    if (!(std::isfinite(nu) && nu > 0)) throw std::runtime_error(runLabel + ": retained molecular nu is invalid");

    PrimitiveMesh mesh;
    mesh.read(caseDir + "/constant/polyMesh");
    FvGeometry geometry;
    geometry.build(mesh);
    const std::vector<FvPatch> patches = buildPatches(mesh, geometry);
    const FieldData<vector> U = readField<vector>(caseDir + "/2000/U");
    const FieldData<scalar> p = readField<scalar>(caseDir + "/2000/p");
    const FieldData<scalar> k = readField<scalar>(caseDir + "/2000/k");
    const FieldData<scalar> omega = readField<scalar>(caseDir + "/2000/omega");
    const FieldData<scalar> nut = readField<scalar>(caseDir + "/2000/nut");
    const FieldData<scalar> nutStart = readField<scalar>(caseDir + "/0/nut");
    const FieldData<scalar> ofY = readField<scalar>(caseDir + "/2000/yPlus");
    const std::vector<vector> cellU = internalValues(U, mesh.nCells());
    const std::vector<scalar> cellK = internalValues(k, mesh.nCells());
    const std::vector<std::vector<scalar>> y = nearWallDist(mesh, geometry, patches);

    // Exercise the same device input assembly used by simpleFoam after a completed step. This is deliberately
    // a no-solve construction from the retained 2000 fields: the old nutBoundary() accessor was empty/stale here,
    // while nutBoundaryAtSample() re-evaluates the active wall treatment into the force workspace. The retained
    // yPlus comparison below uses this returned sample-time nut rather than copying 2000/nut into the input.
    GeometricField<vector> solverU = buildField<vector>(U, patches, mesh.nCells());
    GeometricField<scalar> solverP = buildField<scalar>(p, patches, mesh.nCells());
    GeometricField<scalar> solverK = buildField<scalar>(k, patches, mesh.nCells());
    GeometricField<scalar> solverOmega = buildField<scalar>(omega, patches, mesh.nCells());
    GeometricField<scalar> solverNut = buildField<scalar>(nut, patches, mesh.nCells());
    solverU.evaluateBoundary();
    solverP.evaluateBoundary();
    solverK.evaluateBoundary();
    solverOmega.evaluateBoundary();
    solverNut.evaluateBoundary();
    const SurfaceScalarField phi = fvc::flux(solverU, mesh, geometry, patches);
    DeviceSimpleControls solverCtl;
    solverCtl.nu = nu;
    solverCtl.turbulent = true;
    solverCtl.sst = true;
    solverCtl.useGraph = false;
    DeviceSimpleSolver deviceSolver(mesh, geometry, patches, solverU, solverP, phi, solverCtl,
                                    &solverK, &solverOmega, &solverNut);
    const std::vector<scalar> sampleNut = deviceSolver.nutBoundaryAtSample();
    std::size_t expectedBoundary = 0;
    for (const FvPatch& pch : patches)
        if (!isCoupledInterfaceType(pch.type)) expectedBoundary += static_cast<std::size_t>(pch.size);
    if (sampleNut.size() != expectedBoundary)
        throw std::runtime_error(runLabel + ": sample-time nut assembly returned "
            + std::to_string(sampleNut.size()) + " values; expected " + std::to_string(expectedBoundary));
    for (scalar value : sampleNut)
        if (!std::isfinite(value)) throw std::runtime_error(runLabel + ": sample-time nut assembly is non-finite");
    std::printf("%s solver sample-time nut assembly: %zu non-coupled boundary faces from active device path\n",
                runLabel.c_str(), sampleNut.size());

    // Validate the entire retained boundary layout before selecting the two required wall patches. A field with
    // a different patch count/order is not comparable even if body/lowerWall happen to have the same face count.
    for (const FvPatch& p : patches)
        (void)boundaryValues(ofY, p.name, p.size);

    int failures = 0;
    std::size_t comparedPatches = 0;
    std::size_t flat = 0;
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const FvPatch& p = patches[pi];
        if (isCoupledInterfaceType(p.type)) continue;
        const std::size_t patchFlat = flat;
        flat += static_cast<std::size_t>(p.size);
        if (p.type != "wall") continue;
        const auto* startNut = findYPlusPatchData(nutStart, p);
        if (!startNut) throw std::runtime_error(runLabel + ": no 0/nut wall config for '" + p.name + "'");
        const YPlusWallConfig wall = readYPlusWallConfig(*startNut);
        if (wall.path != YPlusWallPath::Nutk)
            throw std::runtime_error(runLabel + ": retained wall patch '" + p.name + "' is not nutkWallFunction");
        const std::vector<vector> faceU = boundaryValues(U, p.name, p.size);
        const std::vector<scalar> faceNut = boundaryValues(nut, p.name, p.size);
        const std::vector<scalar> of = boundaryValues(ofY, p.name, p.size);
        if (static_cast<label>(of.size()) != p.size)
            throw std::runtime_error(runLabel + ": yPlus patch face count mismatch for '" + p.name + "'");
        YPlusPatchInput input;
        input.name = p.name;
        input.globalStart = p.start;
        input.wall = wall;
        input.faces.reserve(static_cast<std::size_t>(p.size));
        for (label i = 0; i < p.size; ++i)
        {
            const label c = p.faceCells[static_cast<std::size_t>(i)];
            YPlusFaceInput f;
            f.area = p.magSf[static_cast<std::size_t>(i)];
            f.y = y[pi][static_cast<std::size_t>(i)];
            f.deltaCoeff = p.deltaCoeffs[static_cast<std::size_t>(i)];
            f.cellU = cellU[static_cast<std::size_t>(c)];
            f.wallU = faceU[static_cast<std::size_t>(i)];
            f.normal = p.nf[static_cast<std::size_t>(i)];
            f.k = cellK[static_cast<std::size_t>(c)];
            f.wallNut = sampleNut[patchFlat + static_cast<std::size_t>(i)];
            input.faces.push_back(f);
        }
        scalar maxNutDelta = 0;
        for (label i = 0; i < p.size; ++i)
            maxNutDelta = std::fmax(maxNutDelta,
                std::fabs(sampleNut[patchFlat + static_cast<std::size_t>(i)] - faceNut[static_cast<std::size_t>(i)]));
        std::printf("%s patch %-10s solver sample nut vs retained 2000/nut maxAbs=%.9e\n",
                    runLabel.c_str(), p.name.c_str(), (double)maxNutDelta);
        const YPlusSample brae = evaluateYPlus(2000, 2000, {input}, nu);
        std::vector<scalar> cf;
        for (const auto& v : brae.faces) cf.push_back(v.yPlus);
        if (cf.size() != of.size()) throw std::runtime_error(runLabel + ": face ordering/count mismatch for '" + p.name + "'");

        scalar maxAbs = 0, maxRel = 0, ofMean = 0, braeMean = 0;
        std::size_t relativeCount = 0;
        for (std::size_t i = 0; i < of.size(); ++i)
        {
            const scalar d = cf[i] - of[i];
            maxAbs = std::fmax(maxAbs, std::fabs(d));
            if (std::fabs(of[i]) > kNearZero)
            {
                maxRel = std::fmax(maxRel, std::fabs(d)/std::fabs(of[i]));
                ++relativeCount;
            }
        }
        ofMean = areaMean(of, p);
        braeMean = areaMean(cf, p);
        const scalar signedAbs = braeMean - ofMean;
        const bool signedPctAvailable = std::fabs(ofMean) > kNearZero;
        const scalar signedPct = signedPctAvailable ? 100*signedAbs/ofMean : 0;
        const scalar ofBand = bandPercent(of, p);
        const scalar braeBand = bandPercent(cf, p);
        std::printf("%s patch %-10s faces=%d maxAbs=%.9e maxRel=%.9e (near-zero |OF|<=%.1e, n=%zu)\n",
                    runLabel.c_str(), p.name.c_str(), (int)of.size(), (double)maxAbs, (double)maxRel,
                    (double)kNearZero, relativeCount);
        if (signedPctAvailable)
            std::printf("  areaMean OpenFOAM=%.12g Brae=%.12g signedAbs=%.12g signedPct=%.12g%% band OpenFOAM=%.9g%% Brae=%.9g%%\n",
                        (double)ofMean, (double)braeMean, (double)signedAbs, (double)signedPct,
                        (double)ofBand, (double)braeBand);
        else
            std::printf("  areaMean OpenFOAM=%.12g Brae=%.12g signedAbs=%.12g signedPct=n/a (|OF|<=%.1e) band OpenFOAM=%.9g%% Brae=%.9g%%\n",
                        (double)ofMean, (double)braeMean, (double)signedAbs, (double)kNearZero,
                        (double)ofBand, (double)braeBand);
        if (maxAbs > kAbsTol || maxRel > kRelTol) ++failures;
        if (requiredPatch(p.name)) ++comparedPatches;
    }
    if (comparedPatches != 2)
    {
        std::printf("%s: required body and lowerWall were not both compared\n", runLabel.c_str());
        ++failures;
    }
    return failures;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::fprintf(stderr, "usage: %s <issue37-fresh-case> <issue42-fresh-case>\n", argv[0]);
        return 2;
    }
    std::printf("predeclared acceptance: abs<=%.1e and max relative<=%.1e; near-zero |OpenFOAM|<=%.1e\n",
                (double)kAbsTol, (double)kRelTol, (double)kNearZero);
    try
    {
        const int a = compareCase(argv[1], "Issue37");
        const int b = compareCase(argv[2], "Issue42");
        std::printf("retained yPlus comparison: %s\n", (a || b) ? "FAIL" : "PASS");
        return (a || b) ? 1 : 0;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "retained yPlus comparison refused/failed: %s\n", e.what());
        return 1;
    }
}
