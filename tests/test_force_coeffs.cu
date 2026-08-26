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
#include <cmath>
#include <filesystem>
#include <fstream>
#include <initializer_list>
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

static ForceCoeffsConfig readCaseForceConfig(const std::string& caseDir)
{
    const std::string standalone = caseDir + "/system/forceCoeffs";
    if (fs::exists(standalone))
        return readForceCoeffsConfig("forceCoeffs", readDict(standalone));

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

static scalar readOpenFoamCd(const std::string& path, scalar wantedTime)
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
            return cd;
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
    GeometricField<vector> U = buildField<vector>(readField<vector>(fieldDir + "/U"), fvp, m.nCells());
    GeometricField<scalar> p = buildField<scalar>(readField<scalar>(fieldDir + "/p"), fvp, m.nCells());
    U.evaluateBoundary();
    p.evaluateBoundary();

    std::vector<scalar> k;
    const std::string kPath = fieldDir + "/k";
    if (fs::exists(kPath))
    {
        const FieldData<scalar> kfd = readField<scalar>(kPath);
        k = kfd.internalUniform ? std::vector<scalar>(m.nCells(), kfd.internalUniformValue) : kfd.internalField;
    }
    std::vector<scalar> nutBnd(forceBoundaryFaceCount(fvp), scalar(0));
    if (fs::exists(fieldDir + "/nut"))
    {
        GeometricField<scalar> nut = buildField<scalar>(readField<scalar>(fieldDir + "/nut"), fvp, m.nCells());
        nut.evaluateBoundary();
        nutBnd = flattenNonCoupled(nut, fvp);
    }
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
    const scalar ofCd = readOpenFoamCd(coefficientPath, std::stod(timeName));
    const scalar cdRel = std::fabs(brae.Cd - ofCd) / std::max(scalar(1), std::fabs(ofCd));
    std::printf("direct retained OpenFOAM forceCoeffs: time=%s Brae Cd=%.17g OF Cd=%.17g rel=%.3e\n",
                timeName.c_str(), brae.Cd, ofCd, cdRel);
    std::printf("direct device/host force oracle maxRel=%.3e\n", oracleErr);
    // This is a direct same-mesh/same-field coefficient gate, not a copied expected value. The gate is
    // deliberately the existing force-validation tolerance; it is not adjusted toward a known Cd.
    const bool ok = oracleErr <= 5e-10 && cdRel <= 5e-3;
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
