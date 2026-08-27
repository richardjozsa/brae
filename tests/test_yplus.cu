// Focused host-side yPlus contract tests. The same formula helpers are used by simpleFoam and by the
// retained OpenFOAM comparison tool, so these tests exercise the numerical/output contract without requiring
// a CFD solve.
#include "brae_yplus.cuh"
#include "brae_time.cuh"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <unistd.h>

using namespace brae;

namespace {

int failures = 0;

void check(const char* what, bool ok, const std::string& detail = "")
{
    if (ok) return;
    std::printf("  FAIL %s%s\n", what, detail.empty() ? "" : (" [" + detail + "]").c_str());
    ++failures;
}

void near(const char* what, scalar got, scalar want, scalar tol = 1e-12)
{
    check(what, std::isfinite(got) && std::fabs(got - want) <= tol,
          "got=" + std::to_string(got) + " want=" + std::to_string(want));
}

template <typename F>
bool throws(F&& fn)
{
    try { fn(); }
    catch (const std::exception&) { return true; }
    return false;
}

template <typename F>
std::string captureStderr(const std::string& path, F&& fn)
{
    std::fflush(stderr);
    const int saved = dup(fileno(stderr));
    if (!std::freopen(path.c_str(), "w", stderr)) return {};
    fn();
    std::fflush(stderr);
    dup2(saved, fileno(stderr));
    close(saved);
    std::ifstream in(path);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

YPlusFaceInput face(scalar area, scalar wallUx)
{
    YPlusFaceInput f;
    f.area = area;
    f.y = 0.1;
    f.deltaCoeff = 100.0;
    f.cellU = vector{0, 0, 0};
    f.wallU = vector{wallUx, 0, 0};
    f.normal = vector{0, 1, 0};
    f.k = 400.0;
    f.wallNut = 0.0;
    return f;
}

YPlusWallConfig config(YPlusWallPath path)
{
    YPlusWallConfig c;
    c.path = path;
    c.type = "test";
    return c;
}

scalar referenceSpalding(scalar magUp, scalar magGrad, scalar y, scalar nu, scalar nut,
                         scalar kappa, scalar E, int maxIter, scalar tolerance)
{
    scalar ut = std::sqrt((nut + nu) * magGrad);
    scalar err = 0;
    int iter = 0;
    do
    {
        const scalar ku = std::fmin(kappa * magUp / ut, scalar(50));
        const scalar fk = std::exp(ku) - 1 - ku * (1 + 0.5*ku);
        const scalar ff = -ut*y/nu + magUp/ut + (fk - ku*ku*ku/6)/E;
        const scalar df = y/nu + magUp/(ut*ut) + ku*fk/(E*ut);
        const scalar next = ut + ff/df;
        err = std::fabs((ut-next)/ut);
        ut = next;
    } while (ut > 1e-300 && err > tolerance && ++iter < maxIter);
    return y * std::fmax(0.0, ut) / nu;
}

scalar referenceNutU(scalar magUp, scalar magGrad, scalar y, scalar nu, scalar nuEff, scalar kappa, scalar E)
{
    scalar ypl = 11;
    for (int i = 0; i < 10; ++i) ypl = std::log(std::fmax(E*ypl, scalar(1))) / kappa;
    scalar yp = ypl;
    const scalar kappaRe = kappa * magUp * y / nu;
    for (int i = 0; i < 10; ++i)
    {
        const scalar last = yp;
        yp = (kappaRe + yp) / (1 + std::log(std::fmax(E*yp, scalar(1e-300))));
        if (std::fabs((yp-last)/ypl) <= .01) break;
    }
    if (yp < ypl) yp = y * std::sqrt(nuEff * magGrad) / nu;
    return std::fmax(0.0, yp);
}

scalar referenceBlended(scalar magUp, scalar magGrad, scalar y, scalar nu, scalar nut,
                        scalar kappa, scalar E, scalar n)
{
    scalar ut = std::sqrt((nut+nu)*magGrad), error = 1e300;
    for (int i = 0; i < 10 && error > .001; ++i)
    {
        const scalar yp = y*ut/nu;
        const scalar uVis = magUp/yp;
        const scalar uLog = kappa*magUp/std::log(std::fmax(E*yp, scalar(1+.0001)));
        const scalar next = std::pow(std::pow(uVis,n)+std::pow(uLog,n), 1/n);
        error = std::fabs(ut-next)/(ut+1e-300);
        ut = .5*(ut+next);
    }
    return y*std::fmax(0.0,ut)/nu;
}

} // namespace

int main()
{
    const scalar nu = 0.01;

    // Formula paths: nutk inertial and viscous branches, generic wall, Spalding, nutU and blended nutU.
    {
        const auto f = face(1, 1);
        const auto c = config(YPlusWallPath::Nutk);
        near("nutk inertial formula", computeYPlus(c, f, nu),
             std::pow(.09, .25) * f.y * std::sqrt(f.k) / nu, 1e-12);
        auto fv = f;
        fv.k = 0;
        near("nutk viscous formula", computeYPlus(c, fv, nu),
             fv.y * std::sqrt(nu * (vectorMag(fv.wallU-fv.cellU)*fv.deltaCoeff)) / nu, 1e-12);
    }
    {
        auto f = face(1, 1);
        f.k = -1;
        near("generic wall formula", computeYPlus(config(YPlusWallPath::GenericWall), f, nu),
             f.y * std::sqrt(nu * (vectorMag(f.wallU-f.cellU)*f.deltaCoeff)) / nu, 1e-12);
        near("Spalding exact Newton formula", computeYPlus(config(YPlusWallPath::Spalding), face(1, 1), nu),
             referenceSpalding(1, 100, .1, nu, 0, .41, 9.8, 10, .01), 1e-12);
        near("nutU exact fixed point formula", computeYPlus(config(YPlusWallPath::NutU), face(1, 1), nu),
             referenceNutU(1, 100, .1, nu, nu, .41, 9.8), 1e-12);
        const vector du{1, 3, 0};
        f.wallU = du;
        f.normal = vector{0, 1, 0};
        f.k = -1;
        near("nutUBlended uses tangential velocity", computeYPlus(config(YPlusWallPath::Blended), f, nu),
             referenceBlended(1, 100, .1, nu, 0, .41, 9.8, 4), 5e-6);
        check("unsupported rough wall function is refused",
              throws([] { (void)yPlusWallPath("nutkRoughWallFunction"); }));
    }

    // Zero U/gradient is a finite non-negative zero for every supported path.
    for (const auto path : {YPlusWallPath::GenericWall, YPlusWallPath::Nutk,
                            YPlusWallPath::Spalding, YPlusWallPath::NutU, YPlusWallPath::Blended})
    {
        auto f = face(1, 0);
        f.k = (path == YPlusWallPath::Nutk) ? 0 : -1;
        near("zero velocity/gradient", computeYPlus(config(path), f, nu), 0);
    }
    check("near-zero viscosity is refused", throws([&] { (void)computeYPlus(config(YPlusWallPath::GenericWall), face(1, 1), 0); }));
    check("negative area is refused", throws([&] { auto f=face(-1, 1); (void)computeYPlus(config(YPlusWallPath::GenericWall), f, nu); }));

    // Wall-function coefficients must survive field parsing, but identically named entries on an unrelated
    // scalar/vector patch must continue through the ordinary skip path instead of changing yPlus defaults.
    {
        const std::string path = "/tmp/brae_yplus_wall_coeffs_" + std::to_string(static_cast<long long>(getpid()));
        {
            std::ofstream out(path);
            out << "FoamFile { version 2.0; format ascii; class volScalarField; object nut; }\n"
                   "dimensions [0 2 -1 0 0 0 0];\ninternalField uniform 0;\n"
                   "boundaryField { body { type nutUBlendedWallFunction; Cmu 0.123; kappa 0.37; E 12.5; "
                   "n 3; maxIter 17; tolerance 0.002; value uniform 0; } }\n";
        }
        const FieldData<scalar> parsed = readField<scalar>(path);
        const YPlusWallConfig custom = readYPlusWallConfig(parsed.boundary.front());
        check("non-default wall Cmu reaches YPlusWallConfig", std::fabs(custom.Cmu - 0.123) < 1e-14);
        check("non-default wall kappa reaches YPlusWallConfig", std::fabs(custom.kappa - 0.37) < 1e-14);
        check("non-default wall E reaches YPlusWallConfig", std::fabs(custom.E - 12.5) < 1e-14);
        check("non-default wall n reaches YPlusWallConfig", std::fabs(custom.n - 3.0) < 1e-14);
        check("non-default wall maxIter reaches YPlusWallConfig", custom.maxIter == 17);
        check("non-default wall tolerance reaches YPlusWallConfig", std::fabs(custom.tolerance - 0.002) < 1e-14);
        std::filesystem::remove(path);

        {
            std::ofstream out(path);
            out << "FoamFile { version 2.0; format ascii; class volScalarField; object k; }\n"
                   "dimensions [0 2 -2 0 0 0 0];\ninternalField uniform 0;\n"
                   "boundaryField { body { type fixedValue; Cmu 0.123; kappa 0.37; E 12.5; "
                   "n 3; maxIter 17; tolerance 0.002; value uniform 0; } }\n";
        }
        const FieldData<scalar> unrelated = readField<scalar>(path);
        check("wall keys on non-nut patch remain defaults",
              std::fabs(unrelated.boundary.front().wallCmu - 0.09) < 1e-14
              && std::fabs(unrelated.boundary.front().wallKappa - 0.41) < 1e-14
              && std::fabs(unrelated.boundary.front().wallE - 9.8) < 1e-14
              && std::fabs(unrelated.boundary.front().wallN - 4.0) < 1e-14
              && unrelated.boundary.front().wallMaxIter == 10
              && std::fabs(unrelated.boundary.front().wallTolerance - 0.01) < 1e-14);
        std::filesystem::remove(path);
    }

    // Aggregation sorts patches and preserves local face order; area weighting and 30-300 band are explicit.
    {
        YPlusPatchInput lower;
        lower.name = "lowerWall";
        lower.globalStart = 10;
        lower.wall = config(YPlusWallPath::GenericWall);
        lower.faces = {face(1, 1), face(3, 100)};
        YPlusPatchInput body;
        body.name = "body";
        body.globalStart = 20;
        body.wall = config(YPlusWallPath::GenericWall);
        body.faces = {face(2, 10)};
        const YPlusSample s = evaluateYPlus(3, 3, {lower, body}, nu);
        check("non-wall patches are absent", s.patches.size() == 2 && s.faces.size() == 3);
        check("deterministic lexical patch ordering", s.patches[0].patch == "body" && s.patches[1].patch == "lowerWall");
        check("deterministic local face ordering", s.faces[1].patch == "lowerWall"
              && s.faces[1].patchLocalFace == 0 && s.faces[1].globalFace == 10
              && s.faces[2].patchLocalFace == 1 && s.faces[2].globalFace == 11);
        // With y=.1, delta=100, nu=.01: yPlus(1)=10 and yPlus(10)=100.
        near("area-weighted mean", s.patches[1].areaMean, 77.5);
        near("area-weighted 30-300 percentage", s.patches[1].percentBand, 75.0);
        near("body band percentage", s.patches[0].percentBand, 100.0);
        check("correct face area retained", s.faces[1].area == 1 && s.faces[2].area == 3);
    }

    // Unsupported cadence is a refusal, and writer output is Brae-owned, deterministic, complete, and collision-safe.
    {
        FoamDict cadenceDict;
        cadenceDict.leaves.emplace_back("executeControl", std::vector<std::string>{"timeStep"});
        cadenceDict.leaves.emplace_back("executeInterval", std::vector<std::string>{"2"});
        cadenceDict.leaves.emplace_back("writeControl", std::vector<std::string>{"timeStep"});
        cadenceDict.leaves.emplace_back("writeInterval", std::vector<std::string>{"3"});
        cadenceDict.leaves.emplace_back("writeFields", std::vector<std::string>{"true"});
        const YPlusCadence cadence = readYPlusCadence("unit", cadenceDict);
        check("timeStep cadence", cadence.executeDue(2, false) && cadence.writeDue(3, false)
              && !cadence.writeDue(2, false));
        check("execute/write cadence is their intersection",
              !cadence.sampleDue(3, false) && cadence.sampleDue(6, false)
              && !cadence.sampleDue(9, false) && cadence.sampleDue(12, false));
        FoamDict writeFields;
        writeFields.leaves.emplace_back("writeFields", std::vector<std::string>{"true"});
        const std::string writeFieldsNotice = captureStderr("/tmp/brae_yplus_writefields.err", [&] {
            (void)readYPlusCadence("writeFieldsUnique", writeFields);
        });
        check("writeFields true is reported", writeFieldsNotice.find("[approximated] functions/writeFieldsUnique") != std::string::npos
              && writeFieldsNotice.find("does not write an OpenFOAM yPlus volScalarField") != std::string::npos,
              writeFieldsNotice);
        FoamDict bad = cadenceDict;
        bad.leaves.emplace_back("executeControl", std::vector<std::string>{"runTime"});
        check("unsupported cadence is refused", throws([&] { (void)readYPlusCadence("bad", bad); }));

        const std::string root = "/tmp/brae_yplus_writer_contract";
        std::filesystem::remove_all(root);
        std::filesystem::create_directories(root + "/postProcessing/yPlus/0");
        const std::string openfoam = root + "/postProcessing/yPlus/0/yPlus.dat";
        std::ofstream(openfoam) << "retained OpenFOAM bytes\n";
        const std::string before = [&] { std::ifstream in(openfoam); return std::string((std::istreambuf_iterator<char>(in)), {}); }();
        {
            YPlusWriter w(root, "unit", "0", nu, "kOmegaSST", "unit formula", cadence);
            YPlusPatchInput p;
            p.name = "body"; p.globalStart = 5; p.wall = config(YPlusWallPath::GenericWall);
            p.faces = {face(1, 1)};
            w.sample(evaluateYPlus(3, 3, {p}, nu));
            w.finish("iteration_limit", 3);
            check("Brae face evidence path", std::filesystem::exists(w.root() + "/faceValues.dat"));
            check("Brae patch summary path", std::filesystem::exists(w.root() + "/patchSummary.dat"));
            check("completion metadata path", std::filesystem::exists(w.metadataPath()));
        }
        std::ifstream in(openfoam);
        const std::string after((std::istreambuf_iterator<char>(in)), {});
        check("pre-existing OpenFOAM yPlus is byte-identical", before == after);
        check("Brae output has stopping metadata", [&] {
            std::ifstream m(root + "/postProcessing/braeYPlus/unit/0/metadata.dat");
            const std::string x((std::istreambuf_iterator<char>(m)), {});
            return x.find("stopping_reason=iteration_limit") != std::string::npos
                && x.find("completed_iteration=3") != std::string::npos
                && x.find("sample_count=1") != std::string::npos
                && x.find("openfoam_yPlus_field_read=false") != std::string::npos;
        }());
        {
            YPlusWriter w(root, "converged", "0", nu, "kOmegaSST", "unit formula", cadence);
            YPlusPatchInput p;
            p.name = "body"; p.globalStart = 5; p.wall = config(YPlusWallPath::GenericWall);
            p.faces = {face(1, 1)};
            w.sample(evaluateYPlus(3, 3, {p}, nu));
            w.finish("converged", 3);
            std::ifstream m(root + "/postProcessing/braeYPlus/converged/0/metadata.dat");
            const std::string x((std::istreambuf_iterator<char>(m)), {});
            check("converged stopping metadata", x.find("stopping_reason=converged") != std::string::npos);
        }
        std::filesystem::create_directories(root + "/postProcessing/braeYPlus/collision/0");
        std::ofstream(root + "/postProcessing/braeYPlus/collision/0/faceValues.dat") << "sentinel\n";
        check("Brae collision fails closed", throws([&] {
            YPlusWriter w(root, "collision", "0", nu, "kOmegaSST", "unit formula", cadence);
        }));
        std::ifstream sentinel(root + "/postProcessing/braeYPlus/collision/0/faceValues.dat");
        check("collision preserves existing bytes", std::string((std::istreambuf_iterator<char>(sentinel)), {}) == "sentinel\n");
        std::filesystem::remove_all(root);
    }

    // One solver-owned status for the known type; an unknown type remains visibly ignored.
    {
        FoamDict functions;
        FoamDict force, unknown;
        force.leaves.emplace_back("type", std::vector<std::string>{"forceCoeffs"});
        unknown.leaves.emplace_back("type", std::vector<std::string>{"mysteryFunctionObject"});
        functions.subs.emplace_back("statusForceUnique", force);
        functions.subs.emplace_back("statusUnknownUnique", unknown);
        const std::vector<FunctionObjectList::OutsideNotice> outside = {
            {"forceCoeffs", "solver-owned after each completed SIMPLE iteration; not OpenFOAM-identical"}
        };
        const std::string out = captureStderr("/tmp/brae_yplus_status.err", [&] {
            FunctionObjectList list;
            list.read(&functions, {}, {"forceCoeffs"}, outside);
        });
        check("forceCoeffs has one truthful solver-owned status",
              out.find("[solver-owned] functions/statusForceUnique") != std::string::npos
              && out.find("[ignored] functions/statusForceUnique") == std::string::npos
              && out.find("after each completed SIMPLE iteration") != std::string::npos, out);
        check("unknown function object remains ignored",
              out.find("[ignored] functions/statusUnknownUnique") != std::string::npos, out);
    }

    std::printf("yplus: %d failures\n", failures);
    return failures ? 1 : 0;
}
