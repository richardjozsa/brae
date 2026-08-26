// ForceCoeffs contract tests. The first part covers host arithmetic and parsing. The second part runs the
// actual device wall-face reduction on a real cyclic mesh and compares it with the host wallForces oracle.
// With three arguments this executable performs the separate retained-OpenFOAM coefficient.dat comparison.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "cyclic_field.cuh"
#include "foam_field_reader.cuh"
#include "foam_dict.cuh"
#include "forces.cuh"
#include "device_force.cuh"
#include "device_mesh.cuh"
#include "device_boundary.cuh"
#include "cyclic_interface.cuh"
#include "acmi_area_scaling.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace brae;
namespace fs = std::filesystem;

static void leaf(FoamDict& d, const std::string& key, std::initializer_list<std::string> values)
{
    d.leaves.emplace_back(key, std::vector<std::string>(values));
}

static FoamDict config()
{
    FoamDict d;
    leaf(d, "patches", {"upperWall", "lowerWall"});
    leaf(d, "rho", {"rhoInf"});
    leaf(d, "rhoInf", {"2"});
    leaf(d, "magUInf", {"4"});
    leaf(d, "Aref", {"5"});
    leaf(d, "lRef", {"10"});
    leaf(d, "liftDir", {"(", "0", "1", "0", ")"});
    leaf(d, "dragDir", {"(", "1", "0", "0", ")"});
    leaf(d, "pitchAxis", {"(", "0", "0", "1", ")"});
    leaf(d, "CofR", {"(", "1", "2", "3", ")"});
    leaf(d, "pRef", {"7"});
    return d;
}

static DeviceForceSelection makeSelection(const std::vector<FvPatch>& patches, const FvGeometry& g,
                                          const std::vector<std::string>& names)
{
    std::vector<label> boundaryIndex;
    std::vector<scalar> cfx, cfy, cfz;
    label flat = 0;
    for (const FvPatch& p : patches)
    {
        if (isCoupledInterfaceType(p.type)) continue;
        for (label i = 0; i < p.size; ++i)
        {
            if (forcePatchSelected(p, names))
            {
                const vector c = g.Cf()[p.start + i];
                boundaryIndex.push_back(flat);
                cfx.push_back(c.x);
                cfy.push_back(c.y);
                cfz.push_back(c.z);
            }
            ++flat;
        }
    }
    DeviceForceSelection out;
    out.n = static_cast<int>(boundaryIndex.size());
    out.boundaryIndex.copyFrom(boundaryIndex);
    out.cfx.copyFrom(cfx);
    out.cfy.copyFrom(cfy);
    out.cfz.copyFrom(cfz);
    return out;
}

static std::vector<scalar> flattenNonCoupled(const GeometricField<scalar>& f, const std::vector<FvPatch>& patches)
{
    std::vector<scalar> out;
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
        if (!isCoupledInterfaceType(patches[pi].type))
            out.insert(out.end(), f.boundary[pi]->value().begin(), f.boundary[pi]->value().end());
    return out;
}

static scalar forceMaxRelative(const ForceResult& a, const ForceResult& b)
{
    const vector av[4] = {a.pressure, a.viscous, a.momentP, a.momentV};
    const vector bv[4] = {b.pressure, b.viscous, b.momentP, b.momentV};
    scalar worst = 0;
    for (int k = 0; k < 4; ++k)
    {
        const scalar ax[3] = {av[k].x, av[k].y, av[k].z};
        const scalar bx[3] = {bv[k].x, bv[k].y, bv[k].z};
        for (int i = 0; i < 3; ++i)
            worst = std::max(worst, std::fabs(ax[i] - bx[i]) / std::max(scalar(1), std::fabs(bx[i])));
    }
    return worst;
}

static int runCyclicDeviceOracle(const std::string& caseDir)
{
    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const std::vector<CyclicInterface> cyclics = buildCyclicInterfaces(m, g, fvp);
    if (cyclics.empty())
        throw std::runtime_error("forceCoeffs device oracle requires a cyclic fixture");

    const label nC = m.nCells();
    std::vector<vector> Uc(nC);
    std::vector<scalar> pc(nC);
    for (label c = 0; c < nC; ++c)
    {
        const vector C = g.C()[c];
        Uc[c] = {0.3 + 0.2*C.y + 0.01*C.x, -0.1 + 0.05*C.y, 0.02*C.x};
        pc[c] = 0.7*C.x - 0.4*C.y + 0.11*C.z;
    }
    GeometricField<vector> U = buildCyclicField<vector>(Uc, fvp, cyclics, true);
    GeometricField<scalar> p = buildCyclicField<scalar>(pc, fvp, cyclics);

    // The nut values deliberately follow DeviceBoundary order. Since the fixture puts two cyclic patches
    // before the wall, the old all-patches host offset indexed past this vector and could not pass this test.
    std::vector<scalar> nutWall;
    nutWall.reserve(forceBoundaryFaceCount(fvp));
    for (std::size_t i = 0; i < forceBoundaryFaceCount(fvp); ++i)
        nutWall.push_back(0.013 + 0.00017*static_cast<scalar>(i));
    const std::vector<std::string> walls{"walls"};
    const ForceResult host = wallForces(U, p, {}, 0.01, m, g, fvp, walls, 1.2, 0.03,
                                        vector{0.02, -0.01, 0.04}, 0.09, 0.41, 9.8, &nutWall);

    DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    const DeviceVectorBoundary dbU = buildDeviceVectorBoundary(U, fvp, g);
    const DeviceBoundary dbP = buildDeviceBoundary(p, fvp, g);
    DeviceCyclic dcyc = buildDeviceCyclic(cyclics, g, fvp);
    DeviceForceSelection selection = makeSelection(fvp, g, walls);
    if (selection.n == 0)
        throw std::runtime_error("forceCoeffs device oracle selected no wall faces");

    std::vector<scalar> ux(nC), uy(nC), uz(nC), pCell(nC);
    for (label c = 0; c < nC; ++c)
    {
        ux[c] = Uc[c].x; uy[c] = Uc[c].y; uz[c] = Uc[c].z; pCell[c] = pc[c];
    }
    DeviceBuffer<scalar> dUx(ux), dUy(uy), dUz(uz), dp(pCell), dNut(nutWall);
    const DeviceForceResult device = deviceWallForceReduce(
        dm, dbU, dbP, dUx, dUy, dUz, dp, dNut, selection, 0.01, 1.2, 0.03,
        vector{0.02, -0.01, 0.04}, &dcyc, nullptr);
    const ForceResult deviceHost{device.pressure, device.viscous, device.momentP, device.momentV};
    const scalar err = forceMaxRelative(deviceHost, host);
    std::printf("cyclic deviceWallForceReduce vs host wallForces: selected=%d cyclicFaces=%d maxRel=%.3e\n",
                selection.n, dcyc.n, err);
    if (err > 5e-10)
    {
        std::printf("FAIL\n");
        return 1;
    }
    std::printf("cyclic device wall-force oracle: OK\n");
    return 0;
}

static std::string forceCoeffsDictionaryText()
{
    return "type forceCoeffs;\n"
           "patches (body);\n"
           "rho rhoInf;\n"
           "rhoInf 1.0;\n"
           "magUInf 1.0;\n"
           "Aref 0.112032;\n"
           "lRef 1.04;\n"
           "liftDir (0 0 1);\n"
           "dragDir (1 0 0);\n"
           "pitchAxis (0 1 0);\n"
           "CofR (-0.502 0 0);\n";
}

static void writeTextFile(const fs::path& path, const std::string& text)
{
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot create parser test fixture '" + path.string() + "'");
    out << text;
    if (!out) throw std::runtime_error("cannot write parser test fixture '" + path.string() + "'");
}

static ForceCoeffsConfig readStandaloneForceConfig(const FoamDict& file)
{
    const FoamDict* selected = nullptr;
    std::string selectedName;
    for (const auto& s : file.subs)
    {
        const bool hasType = s.second.found("type");
        const std::string type = hasType ? s.second.wordOr("type", "") : "<missing>";
        if (s.first == "forceCoeffs" && type != "forceCoeffs")
            throw std::runtime_error("standalone forceCoeffs object '" + s.first + "' has type '" + type
                                     + "', expected 'forceCoeffs'");
        if (type != "forceCoeffs") continue;
        if (selected)
            throw std::runtime_error("standalone forceCoeffs file has more than one forceCoeffs object ('"
                                     + selectedName + "' and '" + s.first + "')");
        selected = &s.second;
        selectedName = s.first;
    }
    if (!selected)
        throw std::runtime_error("standalone forceCoeffs file has no forceCoeffs object with type 'forceCoeffs'");
    return readForceCoeffsConfig(selectedName, *selected);
}

static ForceCoeffsConfig readCaseForceConfig(const std::string& caseDir)
{
    const std::string standalone = caseDir + "/system/forceCoeffs";
    if (fs::exists(standalone))
        return readStandaloneForceConfig(readDict(standalone));

    const FoamDict control = readDict(caseDir + "/system/controlDict");
    const FoamDict* funcs = control.subDict("functions");
    if (!funcs)
        throw std::runtime_error("retained OpenFOAM case has no functions dictionary");
    const FoamDict* selected = nullptr;
    std::string name;
    for (const auto& s : funcs->subs)
        if (s.second.wordOr("type", "") == "forceCoeffs")
        {
            if (selected) throw std::runtime_error("retained case has more than one forceCoeffs object");
            selected = &s.second;
            name = s.first;
        }
    if (!selected)
        throw std::runtime_error("retained OpenFOAM case has no forceCoeffs object");
    return readForceCoeffsConfig(name, *selected);
}

static int runForceConfigParserTests()
{
    const fs::path root = fs::temp_directory_path()
        / ("brae-force-coeffs-parser-"
           + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(root / "system");
    const fs::path standalone = root / "system/forceCoeffs";
    const fs::path control = root / "system/controlDict";
    int fails = 0;
    auto check = [&](bool ok, const char* name) {
        std::printf("  %-44s %s\n", name, ok ? "OK" : "FAIL");
        if (!ok) ++fails;
    };
    auto throwsContaining = [&](const std::string& needle) {
        try
        {
            (void)readCaseForceConfig(root.string());
        }
        catch (const std::exception& e)
        {
            return std::string(e.what()).find(needle) != std::string::npos;
        }
        return false;
    };

    writeTextFile(standalone, "forceCoeffs\n{\n" + forceCoeffsDictionaryText() + "}\n");
    const ForceCoeffsConfig named = readCaseForceConfig(root.string());
    check(named.name == "forceCoeffs" && named.patches == std::vector<std::string>{"body"}
              && std::fabs(named.Aref - 0.112032) < 1e-15,
          "standalone named forceCoeffs dictionary");

    std::error_code ec;
    fs::remove(standalone, ec);
    writeTextFile(control, "functions\n{\n  coeffs\n  {\n" + forceCoeffsDictionaryText() + "}\n}\n");
    const ForceCoeffsConfig inlineConfig = readCaseForceConfig(root.string());
    check(inlineConfig.name == "coeffs" && inlineConfig.patches == std::vector<std::string>{"body"},
          "inline functions forceCoeffs dictionary");

    fs::remove(control, ec);
    writeTextFile(standalone, "other\n{\n  type forces;\n}\n");
    check(throwsContaining("no forceCoeffs object"), "standalone missing forceCoeffs object fails");

    writeTextFile(standalone,
                  "first\n{\n" + forceCoeffsDictionaryText() + "}\n"
                  "second\n{\n" + forceCoeffsDictionaryText() + "}\n");
    check(throwsContaining("more than one forceCoeffs object"), "standalone duplicate objects fail");

    writeTextFile(standalone, "forceCoeffs\n{\n  type forces;\n}\n");
    check(throwsContaining("expected 'forceCoeffs'"), "standalone malformed object fails");

    fs::remove_all(root, ec);
    return fails;
}

struct OpenFoamCoefficientRow
{
    scalar time = 0;
    scalar Cd = 0;
};

static OpenFoamCoefficientRow readOpenFoamCd(const std::string& path, scalar wantedTime)
{
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot read retained OpenFOAM coefficient file '" + path + "'");
    std::string line;
    while (std::getline(in, line))
    {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream row(line);
        scalar time = 0, cd = 0;
        if (row >> time >> cd && std::fabs(time - wantedTime) <= 1e-9)
            return {time, cd};
    }
    throw std::runtime_error("retained OpenFOAM coefficient file has no row at requested time "
                             + std::to_string(wantedTime));
}

static int runRetainedOpenFoamComparison(const std::string& caseDir, const std::string& timeName,
                                         const std::string& coefficientPath)
{
    const ForceCoeffsConfig cfg = readCaseForceConfig(caseDir);
    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    std::vector<FvPatch> fvp;
    std::vector<AMIInterface> amis;
    buildGeometryPatchesAndAMI(m, g, fvp, amis);
    const std::vector<CyclicInterface> cyclics = buildCyclicInterfaces(m, g, fvp);
    const std::string fieldDir = caseDir + "/" + timeName;
    const scalar wantedTime = std::stod(timeName);
    const fs::path canonicalCase = fs::weakly_canonical(fs::path(caseDir));
    const fs::path canonicalCoefficient = fs::weakly_canonical(fs::path(coefficientPath));
    const std::string coefficientRelative = fs::relative(canonicalCoefficient, canonicalCase).generic_string();
    if (coefficientRelative.empty() || coefficientRelative == ".."
        || coefficientRelative.rfind("../", 0) == 0)
        throw std::runtime_error("retained OpenFOAM coefficient file is outside the retained case: '"
                                 + coefficientPath + "'");
    const std::string timeMetadataPath = fieldDir + "/uniform/time";
    if (!fs::exists(timeMetadataPath) && !fs::exists(timeMetadataPath + ".gz"))
        throw std::runtime_error("retained OpenFOAM time metadata is missing for field directory '" + fieldDir + "'");
    const FoamDict timeMetadata = readDict(timeMetadataPath);
    const scalar resolvedTime = timeMetadata.scalarOr("value", std::numeric_limits<scalar>::quiet_NaN());
    if (!std::isfinite(resolvedTime) || std::fabs(resolvedTime - wantedTime) > 1e-9)
        throw std::runtime_error("retained OpenFOAM time metadata does not match requested time '" + timeName + "'");
    auto requireField = [&](const char* fieldName) {
        const std::string path = fieldDir + "/" + fieldName;
        if (!fs::exists(path) && !fs::exists(path + ".gz"))
            throw std::runtime_error("retained OpenFOAM field '" + path + "' is missing");
    };
    for (const char* fieldName : {"U", "p", "k", "nut"}) requireField(fieldName);
    GeometricField<vector> U = buildField<vector>(readField<vector>(fieldDir + "/U"), fvp, m.nCells());
    GeometricField<scalar> p = buildField<scalar>(readField<scalar>(fieldDir + "/p"), fvp, m.nCells());
    U.evaluateBoundary();
    p.evaluateBoundary();

    const FieldData<scalar> kfd = readField<scalar>(fieldDir + "/k");
    if (!kfd.internalUniform && static_cast<label>(kfd.internalField.size()) != m.nCells())
        throw std::runtime_error("retained OpenFOAM k field does not match the retained mesh cell count");
    const std::vector<scalar> k = kfd.internalUniform
        ? std::vector<scalar>(m.nCells(), kfd.internalUniformValue) : kfd.internalField;
    std::vector<scalar> nutBnd(forceBoundaryFaceCount(fvp), scalar(0));
    GeometricField<scalar> nut = buildField<scalar>(readField<scalar>(fieldDir + "/nut"), fvp, m.nCells());
    nut.evaluateBoundary();
    nutBnd = flattenNonCoupled(nut, fvp);
    const FoamDict transport = readDict(caseDir + "/constant/transportProperties");
    const scalar nu = transport.scalarOr("nu", 1e-5);
    const ForceResult host = wallForces(U, p, k, nu, m, g, fvp, cfg.patches, cfg.rhoInf, cfg.pRef,
                                        cfg.CofR, 0.09, 0.41, 9.8, &nutBnd);

    DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    const DeviceVectorBoundary dbU = buildDeviceVectorBoundary(U, fvp, g);
    const DeviceBoundary dbP = buildDeviceBoundary(p, fvp, g);
    DeviceCyclic dcyc = buildDeviceCyclic(cyclics, g, fvp);
    DeviceAMI dami = buildDeviceAMI(amis);
    DeviceForceSelection selection = makeSelection(fvp, g, cfg.patches);
    if (selection.n == 0) throw std::runtime_error("retained case forceCoeffs selected no non-coupled faces");
    std::vector<scalar> ux(m.nCells()), uy(m.nCells()), uz(m.nCells()), pc(m.nCells());
    for (label c = 0; c < m.nCells(); ++c)
    {
        ux[c] = U.internal[c].x; uy[c] = U.internal[c].y; uz[c] = U.internal[c].z; pc[c] = p.internal[c];
    }
    DeviceBuffer<scalar> dUx(ux), dUy(uy), dUz(uz), dp(pc), dNut(nutBnd);
    const DeviceForceResult device = deviceWallForceReduce(
        dm, dbU, dbP, dUx, dUy, dUz, dp, dNut, selection, nu, cfg.rhoInf, cfg.pRef, cfg.CofR,
        dcyc.n ? &dcyc : nullptr, dami.n ? &dami : nullptr);
    const ForceResult deviceForce{device.pressure, device.viscous, device.momentP, device.momentV};
    const scalar oracleErr = forceMaxRelative(deviceForce, host);
    const ForceCoeffs brae = forceCoeffs(deviceForce, cfg.dragDir, cfg.liftDir, cfg.pitchAxis,
                                         cfg.rhoInf, cfg.magUInf, cfg.Aref, cfg.lRef);
    const OpenFoamCoefficientRow ofRow = readOpenFoamCd(coefficientPath, wantedTime);
    const scalar signedAbsoluteDifference = brae.Cd - ofRow.Cd;
    // Percentage policy: when |OpenFOAM Cd| <= 1e-12, a percentage is undefined. Report that explicitly and
    // use an absolute tolerance of 1e-12 for the gate; otherwise report the signed (Brae-OpenFOAM)/|OpenFOAM|.
    constexpr scalar cdNearZero = 1e-12;
    constexpr scalar cdPercentageTolerance = 0.5; // existing direct-comparison threshold, in percentage points
    const bool percentageDefined = std::fabs(ofRow.Cd) > cdNearZero;
    const scalar signedPercentageDifference = percentageDefined
        ? 100.0 * signedAbsoluteDifference / std::fabs(ofRow.Cd) : 0.0;
    std::printf("direct retained OpenFOAM forceCoeffs:\n");
    std::printf("  case=%s\n", canonicalCase.string().c_str());
    std::printf("  time=%s field_time=%.17g coefficient_row_time=%.17g\n",
                timeName.c_str(), resolvedTime, ofRow.time);
    std::printf("  fields=U,p,k,nut from %s/%s\n", canonicalCase.string().c_str(), timeName.c_str());
    std::printf("  coefficient_row=%s\n", canonicalCoefficient.string().c_str());
    std::printf("  Brae Cd=%.17g\n", brae.Cd);
    std::printf("  OpenFOAM Cd=%.17g\n", ofRow.Cd);
    std::printf("  signed absolute difference (Brae-OpenFOAM)=%+.17g\n", signedAbsoluteDifference);
    if (percentageDefined)
        std::printf("  signed percentage difference ((Brae-OpenFOAM)/abs(OpenFOAM))=%+.9g%%\n",
                    signedPercentageDifference);
    else
        std::printf("  signed percentage difference ((Brae-OpenFOAM)/abs(OpenFOAM))=undefined "
                    "(|OpenFOAM Cd| <= %.1e)\n", cdNearZero);
    std::printf("  device-versus-host force reduction discrepancy "
                "(max |device-host|/max(1,|host|))=%.3e\n", oracleErr);
    // This is a direct same-mesh/same-field coefficient gate, not a copied expected value. The gate is
    // deliberately the existing force-validation tolerance; it is not adjusted toward a known Cd.
    const bool cdOk = percentageDefined ? std::fabs(signedPercentageDifference) <= cdPercentageTolerance
                                        : std::fabs(signedAbsoluteDifference) <= cdNearZero;
    const bool ok = oracleErr <= 5e-10 && cdOk;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int main(int argc, char** argv)
{
    try
    {
        if (argc == 4)
            return runRetainedOpenFoamComparison(argv[1], argv[2], argv[3]);

        int fails = 0;
        auto check = [&](bool ok, const char* name) {
            std::printf("  %-44s %s\n", name, ok ? "OK" : "FAIL");
            if (!ok) ++fails;
        };

        ForceResult F;
        F.pressure = {8, 4, 0};
        F.viscous  = {2, 6, 0};
        F.momentP  = {0, 0, 10};
        F.momentV  = {0, 0, 10};
        const ForceCoeffs c = forceCoeffs(F, {1,0,0}, {0,1,0}, {0,0,1}, 2, 4, 5, 10);
        check(std::fabs(c.Cd - 0.125) < 1e-14, "rhoInf, magUInf, Aref -> Cd");
        check(std::fabs(c.Cl - 0.125) < 1e-14, "configured lift direction -> Cl");
        check(std::fabs(c.Cm + 0.025) < 1e-14, "derived OF side direction -> Cm");
        const ForceCoeffs scaledDirs = forceCoeffs(F, {2,0,0}, {0,3,0}, {0,0,4}, 2, 4, 5, 10);
        check(std::fabs(scaledDirs.Cd - c.Cd) < 1e-14 && std::fabs(scaledDirs.Cl - c.Cl) < 1e-14,
              "non-unit directions are normalized");
        const ForceCoeffs yDrag = forceCoeffs(F, {0,1,0}, {1,0,0}, {0,0,1}, 2, 4, 5, 10);
        check(std::fabs(yDrag.Cd - c.Cl) < 1e-14, "configured drag direction -> Cd");
        const ForceCoeffs twiceArea = forceCoeffs(F, {1,0,0}, {0,1,0}, {0,0,1}, 2, 4, 10, 10);
        check(std::fabs(twiceArea.Cd - 0.0625) < 1e-14, "configured Aref changes normalization");
        const ForceCoeffs twiceRho = forceCoeffs(F, {1,0,0}, {0,1,0}, {0,0,1}, 4, 4, 5, 10);
        check(std::fabs(twiceRho.Cd - 0.0625) < 1e-14, "rhoInf participates in q normalization");

        FoamDict parsedDict = config();
        leaf(parsedDict, "executeControl", {"timeStep"});
        leaf(parsedDict, "executeInterval", {"1"});
        leaf(parsedDict, "writeControl", {"timeStep"});
        leaf(parsedDict, "writeInterval", {"1"});
        const ForceCoeffsConfig parsed = readForceCoeffsConfig("coeffs", parsedDict);
        check(parsed.rhoName == "rhoInf" && parsed.rhoInf == 2 && parsed.Aref == 5 && parsed.pRef == 7,
              "strict parser consumes rho/rhoInf and pRef");
        check(parsed.patches.size() == 2 && parsed.dragDir.x == 1 && parsed.CofR.z == 3,
              "strict parser consumes patches/directions/CofR");

        FoamDict cadence = config();
        leaf(cadence, "writeControl", {"runTime"});
        bool cadenceFailed = false;
        try { (void)readForceCoeffsConfig("cadence", cadence); }
        catch (const std::exception& e) { cadenceFailed = std::string(e.what()).find("writeControl") != std::string::npos; }
        check(cadenceFailed, "unsupported write cadence fails clearly");

        FvPatch group;
        group.name = "bodyWall";
        group.inGroups = {"aeroWalls"};
        check(forcePatchSelected(group, {"aeroWalls"}), "patchSet group selection");
        check(forcePatchSelected(group, {"body.*"}), "patchSet regex selection");
        check(!forcePatchSelected(group, {"other"}), "patchSet rejects unrelated patch");

        FoamDict missing = config();
        for (auto it = missing.leaves.begin(); it != missing.leaves.end(); ++it)
            if (it->first == "Aref") { missing.leaves.erase(it); break; }
        bool missingFailed = false;
        try { (void)readForceCoeffsConfig("missing", missing); }
        catch (const std::exception& e) { missingFailed = std::string(e.what()).find("'Aref' is missing") != std::string::npos; }
        check(missingFailed, "missing required Aref fails clearly");

        FoamDict unsupported = config();
        for (auto& e : unsupported.leaves) if (e.first == "rho") e.second = {"rho"};
        bool rhoFailed = false;
        try { (void)readForceCoeffsConfig("rhoField", unsupported); }
        catch (const std::exception& e) { rhoFailed = std::string(e.what()).find("unsupported") != std::string::npos; }
        check(rhoFailed, "unsupported rho behavior fails clearly");

        fails += runForceConfigParserTests();
        if (runCyclicDeviceOracle(argc > 1 ? argv[1] : "validation/cyclicChannel") != 0) ++fails;
        std::printf("%s\n", fails ? "FAIL" : "PASS");
        return fails ? 1 : 0;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "test_force_coeffs ERROR: %s\n", e.what());
        return 1;
    }
}
