#pragma once
// Brae-owned simpleFoam yPlus observability.
//
// The formulas in this file are deliberately separate from the momentum wall-nut implementation. OpenFOAM's
// functionObjects::yPlus calls the active nut wall-function's virtual yPlus(), and those methods do NOT all use
// the same estimate. This header is the host-side, auditable equivalent used both by the solver and by the retained
// OpenFOAM comparison tool.

#include "cf_types.cuh"
#include "foam_dict.cuh"
#include "foam_field_reader.cuh"
#include "fv_patch.cuh"
#include "nut_wall_function.cuh"
#include "brae_notice.cuh"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace brae {

enum class YPlusWallPath
{
    GenericWall,
    Nutk,
    Spalding,
    NutU,
    Blended
};

inline const char* yPlusWallPathName(YPlusWallPath p)
{
    switch (p)
    {
        case YPlusWallPath::GenericWall: return "generic wall: y*sqrt(nuEff*|snGrad(U)|)/nu";
        case YPlusWallPath::Nutk:        return "nutkWallFunction: inertial Cmu^0.25*y*sqrt(k)/nu, viscous y*sqrt(nuEff*|snGrad(U)|)/nu";
        case YPlusWallPath::Spalding:    return "nutUSpaldingWallFunction: Newton Spalding uTau, y*uTau/nu";
        case YPlusWallPath::NutU:        return "nutUWallFunction: log-law fixed-point yPlus, viscous nuEff/snGrad branch";
        case YPlusWallPath::Blended:     return "nutUBlendedWallFunction: tangential binomial n blend uTau, y*uTau/nu";
    }
    return "unknown";
}

inline YPlusWallPath yPlusWallPath(const std::string& type)
{
    if (type == "nutkWallFunction") return YPlusWallPath::Nutk;
    if (type == "nutUSpaldingWallFunction") return YPlusWallPath::Spalding;
    if (type == "nutUWallFunction") return YPlusWallPath::NutU;
    if (type == "nutUBlendedWallFunction") return YPlusWallPath::Blended;
    if (type == "nutLowReWallFunction" || type == "nutkRoughWallFunction"
        || type == "nutURoughWallFunction" || type == "nutUTabulatedWallFunction"
        || type == "atmNutkWallFunction")
        throw std::runtime_error("yPlus: wall-function type '" + type
            + "' is not supported by the exact Brae path; refusing to substitute a smooth-wall formula");
    // OpenFOAM's yPlus falls back to the generic wall expression for an ordinary wall patch whose nut field is not
    // a nutWallFunction-derived field. The caller only supplies geometric wall patches here.
    return YPlusWallPath::GenericWall;
}

inline scalar yPlusWallLam(scalar kappa, scalar E)
{
    if (!(std::isfinite(kappa) && kappa > 0) || !(std::isfinite(E) && E > 0))
        throw std::runtime_error("yPlus: wall-function kappa and E must be finite and > 0");
    scalar ypl = 11.0;
    for (int iter = 0; iter < 10; ++iter)
        ypl = std::log(std::fmax(E * ypl, scalar(1.0))) / kappa;
    if (!(std::isfinite(ypl) && ypl > 0))
        throw std::runtime_error("yPlus: wall-function yPlusLam calculation was non-finite");
    return ypl;
}

struct YPlusWallConfig
{
    std::string type;
    YPlusWallPath path = YPlusWallPath::GenericWall;
    scalar Cmu = 0.09;
    scalar kappa = 0.41;
    scalar E = 9.8;
    scalar n = 4.0;
    int maxIter = 10;
    scalar tolerance = 0.01;
};

inline YPlusWallConfig readYPlusWallConfig(const PatchFieldData<scalar>& p)
{
    YPlusWallConfig c;
    c.type = p.type;
    c.path = yPlusWallPath(p.type);
    c.Cmu = p.wallCmu;
    c.kappa = p.wallKappa;
    c.E = p.wallE;
    c.n = p.wallN;
    c.maxIter = p.wallMaxIter;
    c.tolerance = p.wallTolerance;
    if (!(std::isfinite(c.Cmu) && c.Cmu > 0))
        throw std::runtime_error("yPlus: wall patch '" + p.name + "' has invalid Cmu");
    if (!(std::isfinite(c.kappa) && c.kappa > 0) || !(std::isfinite(c.E) && c.E > 0))
        throw std::runtime_error("yPlus: wall patch '" + p.name + "' has invalid kappa/E");
    if (c.path == YPlusWallPath::Blended && !(std::isfinite(c.n) && c.n > 0))
        throw std::runtime_error("yPlus: wall patch '" + p.name + "' has invalid nutUBlended exponent n");
    if (c.path == YPlusWallPath::Spalding
        && (c.maxIter <= 0 || !(std::isfinite(c.tolerance) && c.tolerance >= 0)))
        throw std::runtime_error("yPlus: wall patch '" + p.name + "' has invalid Spalding maxIter/tolerance");
    return c;
}

struct YPlusFaceInput
{
    scalar area = 0;
    scalar y = 0;
    scalar deltaCoeff = 0;
    vector cellU{};
    vector wallU{};
    vector normal{};
    scalar k = 0;
    scalar wallNut = 0;
};

struct YPlusPatchInput
{
    std::string name;
    label globalStart = 0;
    YPlusWallConfig wall;
    std::vector<YPlusFaceInput> faces;
};

// Keep boundary-field resolution identical to GeometricField: an exact patch entry wins, then the last
// matching group/regex entry wins. yPlus must not silently use a different wall-function configuration when
// the case uses a grouped or regular-expression boundaryField entry.
inline const PatchFieldData<scalar>* findYPlusPatchData(
    const FieldData<scalar>& field, const FvPatch& patch)
{
    for (const auto& b : field.boundary)
        if (b.name == patch.name) return &b;
    const PatchFieldData<scalar>* match = nullptr;
    for (const auto& b : field.boundary)
    {
        bool hit = false;
        for (const auto& group : patch.inGroups)
            if (b.name == group) { hit = true; break; }
        if (!hit)
        {
            try
            {
                const std::regex re = compileFoamRegex(b.name);
                hit = std::regex_match(patch.name, re);
                if (!hit)
                    for (const auto& group : patch.inGroups)
                        if (std::regex_match(group, re)) { hit = true; break; }
            }
            catch (const std::regex_error&) {}
        }
        if (hit) match = &b;
    }
    return match;
}

inline scalar vectorMag(const vector& v)
{
    return std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
}

inline scalar tangentialMag(const vector& v, const vector& n)
{
    const scalar nd = dot(n, v);
    const vector t = v - n * nd;
    return vectorMag(t);
}

inline scalar yPlusSpalding(
    scalar magUp, scalar magGradU, scalar y, scalar nu, scalar wallNut,
    scalar kappa, scalar E, int maxIter, scalar tolerance)
{
    if (!(std::isfinite(magUp) && magUp >= 0) || !(std::isfinite(magGradU) && magGradU >= 0)
        || !(std::isfinite(y) && y >= 0) || !(std::isfinite(nu) && nu > 0)
        || !(std::isfinite(wallNut) && wallNut + nu >= 0))
        throw std::runtime_error("yPlus: invalid Spalding input");
    if (magUp <= 1e-300 || magGradU <= 1e-300 || y <= 0) return 0;

    scalar ut = std::sqrt((wallNut + nu) * magGradU);
    if (ut > 1e-300)
    {
        int iter = 0;
        scalar err = 0;
        do
        {
            const scalar kUu = std::fmin(kappa * magUp / ut, scalar(50));
            const scalar fkUu = std::exp(kUu) - 1.0 - kUu * (1.0 + 0.5*kUu);
            const scalar f = -ut*y/nu + magUp/ut
                           + (1.0/E) * (fkUu - (1.0/6.0)*kUu*kUu*kUu);
            const scalar df = y/nu + magUp/(ut*ut) + (1.0/E)*kUu*fkUu/ut;
            const scalar next = ut + f/df;
            err = std::fabs((ut - next) / ut);
            ut = next;
        }
        while (ut > 1e-300 && err > tolerance && ++iter < maxIter);
    }
    const scalar uTau = std::fmax(0.0, ut);
    const scalar result = y * uTau / nu;
    if (!std::isfinite(result)) throw std::runtime_error("yPlus: Spalding result is non-finite");
    return std::fmax(0.0, result);
}

inline scalar yPlusNutU(
    scalar magUp, scalar magGradU, scalar y, scalar nu, scalar nuEff,
    scalar kappa, scalar E)
{
    if (!(std::isfinite(magUp) && magUp >= 0) || !(std::isfinite(magGradU) && magGradU >= 0)
        || !(std::isfinite(y) && y >= 0) || !(std::isfinite(nu) && nu > 0)
        || !(std::isfinite(nuEff) && nuEff >= 0))
        throw std::runtime_error("yPlus: invalid nutU input");
    if (magUp <= 1e-300 || y <= 0) return 0;
    const scalar ypl = yPlusWallLam(kappa, E);
    const scalar kappaRe = kappa * magUp * y / nu;
    scalar yp = ypl;
    for (int iter = 0; iter < 10; ++iter)
    {
        const scalar last = yp;
        yp = (kappaRe + yp) / (1.0 + std::log(std::fmax(E*yp, scalar(1e-300))));
        if (std::fabs((yp-last)/ypl) <= 0.01) break;
    }
    yp = std::fmax(0.0, yp);
    if (yp < ypl)
        yp = y * std::sqrt(nuEff * magGradU) / nu;
    if (!std::isfinite(yp)) throw std::runtime_error("yPlus: nutU result is non-finite");
    return std::fmax(0.0, yp);
}

inline scalar yPlusBlended(
    scalar magUpTangential, scalar magGradU, scalar y, scalar nu, scalar wallNut,
    scalar kappa, scalar E, scalar n)
{
    if (!(std::isfinite(magUpTangential) && magUpTangential >= 0)
        || !(std::isfinite(magGradU) && magGradU >= 0) || !(std::isfinite(y) && y >= 0)
        || !(std::isfinite(nu) && nu > 0) || !(std::isfinite(wallNut) && wallNut + nu >= 0)
        || !(std::isfinite(n) && n > 0))
        throw std::runtime_error("yPlus: invalid nutUBlended input");
    if (magUpTangential <= 1e-300 || magGradU <= 1e-300 || y <= 0) return 0;

    scalar ut = std::sqrt((wallNut + nu) * magGradU);
    scalar error = std::numeric_limits<scalar>::max();
    for (int iter = 0; iter < 10 && error > 0.001; ++iter)
    {
        const scalar yp = y * ut / nu;
        if (!(yp > 1e-300)) return 0;
        const scalar uTauVis = magUpTangential / yp;
        const scalar uTauLog = kappa * magUpTangential / std::log(std::fmax(E*yp, scalar(1.0 + 1e-4)));
        const scalar utNew = std::pow(std::pow(uTauVis, n) + std::pow(uTauLog, n), 1.0/n);
        error = std::fabs(ut - utNew) / (ut + 1e-300);
        ut = 0.5 * (ut + utNew);
    }
    const scalar result = y * std::fmax(0.0, ut) / nu;
    if (!std::isfinite(result)) throw std::runtime_error("yPlus: nutUBlended result is non-finite");
    return std::fmax(0.0, result);
}

inline scalar computeYPlus(const YPlusWallConfig& wall, const YPlusFaceInput& f, scalar nu)
{
    if (!(std::isfinite(nu) && nu > 0))
        throw std::runtime_error("yPlus: molecular kinematic viscosity must be finite and > 0");
    if (!(std::isfinite(f.area) && f.area > 0) || !(std::isfinite(f.y) && f.y > 0)
        || !(std::isfinite(f.deltaCoeff) && f.deltaCoeff > 0) || !(std::isfinite(f.wallNut)))
        throw std::runtime_error("yPlus: wall face has invalid area, distance, gradient coefficient, or wall nut");
    if (wall.path == YPlusWallPath::Nutk && !(std::isfinite(f.k) && f.k >= 0))
        throw std::runtime_error("yPlus: nutkWallFunction wall face has invalid adjacent-cell k");
    const vector du = f.wallU - f.cellU;
    const scalar magUp = vectorMag(du);
    const scalar magGradU = magUp * f.deltaCoeff;
    const scalar nuEff = nu + f.wallNut;
    if (!(std::isfinite(nuEff) && nuEff >= 0))
        throw std::runtime_error("yPlus: wall effective viscosity is non-finite or negative");

    scalar yp = 0;
    switch (wall.path)
    {
        case YPlusWallPath::GenericWall:
            yp = f.y * std::sqrt(nuEff * magGradU) / nu;
            break;
        case YPlusWallPath::Nutk:
        {
            const scalar inertial = std::pow(wall.Cmu, 0.25) * f.y * std::sqrt(f.k) / nu;
            const scalar lam = yPlusWallLam(wall.kappa, wall.E);
            yp = (lam > inertial) ? f.y * std::sqrt(nuEff * magGradU) / nu : inertial;
            break;
        }
        case YPlusWallPath::Spalding:
            yp = yPlusSpalding(magUp, magGradU, f.y, nu, f.wallNut,
                               wall.kappa, wall.E, wall.maxIter, wall.tolerance);
            break;
        case YPlusWallPath::NutU:
            yp = yPlusNutU(magUp, magGradU, f.y, nu, nuEff, wall.kappa, wall.E);
            break;
        case YPlusWallPath::Blended:
            yp = yPlusBlended(tangentialMag(du, f.normal), magGradU, f.y, nu, f.wallNut,
                              wall.kappa, wall.E, wall.n);
            break;
    }
    if (!(std::isfinite(yp) && yp >= 0))
        throw std::runtime_error("yPlus: non-finite or negative value; refusing to write output");
    return yp;
}

struct YPlusFaceValue
{
    scalar time = 0;
    int iteration = 0;
    std::string patch;
    label patchLocalFace = 0;
    label globalFace = -1;
    scalar area = 0;
    scalar y = 0;
    scalar yPlus = 0;
};

struct YPlusPatchSummary
{
    scalar time = 0;
    int iteration = 0;
    std::string patch;
    label faceCount = 0;
    scalar totalArea = 0;
    scalar areaMean = 0;
    scalar minimum = 0;
    scalar maximum = 0;
    scalar percentBand = 0;
};

struct YPlusSample
{
    std::vector<YPlusFaceValue> faces;
    std::vector<YPlusPatchSummary> patches;
};

inline YPlusSample evaluateYPlus(
    scalar time, int iteration, std::vector<YPlusPatchInput> patches, scalar nu)
{
    std::sort(patches.begin(), patches.end(), [](const YPlusPatchInput& a, const YPlusPatchInput& b)
    { return a.name < b.name; });
    YPlusSample out;
    for (const auto& p : patches)
    {
        if (p.name.empty()) throw std::runtime_error("yPlus: wall patch name is empty");
        YPlusPatchSummary s;
        s.time = time;
        s.iteration = iteration;
        s.patch = p.name;
        s.faceCount = static_cast<label>(p.faces.size());
        if (s.faceCount <= 0) throw std::runtime_error("yPlus: geometric wall patch '" + p.name + "' is empty");
        s.minimum = std::numeric_limits<scalar>::max();
        s.maximum = 0;
        scalar bandArea = 0;
        for (label i = 0; i < s.faceCount; ++i)
        {
            const auto& f = p.faces[static_cast<std::size_t>(i)];
            const scalar yp = computeYPlus(p.wall, f, nu);
            YPlusFaceValue v{time, iteration, p.name, i, p.globalStart + i, f.area, f.y, yp};
            out.faces.push_back(v);
            s.totalArea += f.area;
            s.areaMean += f.area * yp;
            s.minimum = std::fmin(s.minimum, yp);
            s.maximum = std::fmax(s.maximum, yp);
            if (yp >= 30.0 && yp <= 300.0) bandArea += f.area;
        }
        if (!(std::isfinite(s.totalArea) && s.totalArea > 0))
            throw std::runtime_error("yPlus: wall patch '" + p.name + "' has non-positive/non-finite total area");
        s.areaMean /= s.totalArea;
        s.percentBand = 100.0 * bandArea / s.totalArea;
        if (!(std::isfinite(s.areaMean) && std::isfinite(s.minimum) && std::isfinite(s.maximum)
              && std::isfinite(s.percentBand)))
            throw std::runtime_error("yPlus: patch summary for '" + p.name + "' is non-finite");
        out.patches.push_back(s);
    }
    return out;
}

struct YPlusCadence
{
    std::string executeControl = "timeStep";
    scalar executeInterval = 1;
    std::string writeControl = "writeTime";
    scalar writeInterval = 1;

    static void checkInterval(const std::string& object, const char* key, scalar x)
    {
        if (!(std::isfinite(x) && x >= 1 && std::fabs(x - std::round(x)) <= 1e-12))
            throw std::runtime_error("yPlus '" + object + "': " + key
                + " must be a positive integer iteration interval");
    }

    bool executeDue(int iteration, bool globalWrite) const
    {
        if (executeControl == "writeTime") return globalWrite;
        return iteration % static_cast<int>(std::llround(executeInterval)) == 0;
    }

    bool writeDue(int iteration, bool globalWrite) const
    {
        if (writeControl == "writeTime") return globalWrite;
        if (writeControl == "none") return false;
        return iteration % static_cast<int>(std::llround(writeInterval)) == 0;
    }

    // simpleFoam owns this observable outside Time::functionObjects. Its sample is emitted only at the
    // intersection of the configured execute and write schedules; this is deliberately not an OF-identical
    // execute-only path.
    bool sampleDue(int iteration, bool globalWrite) const
    {
        return executeDue(iteration, globalWrite) && writeDue(iteration, globalWrite);
    }
};

inline YPlusCadence readYPlusCadence(const std::string& object, const FoamDict& d)
{
    YPlusCadence c;
    c.executeControl = d.wordOr("executeControl", "timeStep");
    c.executeInterval = d.scalarOr("executeInterval", 1.0);
    c.writeControl = d.wordOr("writeControl", "writeTime");
    c.writeInterval = d.scalarOr("writeInterval", 1.0);
    (void)d.wordListOr("libs", {}); // recognized metadata; Brae never loads an OpenFOAM shared library.
    if (d.found("writeFields"))
    {
        const std::string writeFields = d.wordOr("writeFields", "false");
        if (writeFields == "true" || writeFields == "yes" || writeFields == "on" || writeFields == "1")
            noticeApproximated("functions/" + object,
                "writeFields true requested, but Brae writes only Brae-owned face/patch evidence and does not write an OpenFOAM yPlus volScalarField");
    }
    if (c.executeControl != "timeStep" && c.executeControl != "writeTime")
        throw std::runtime_error("yPlus '" + object + "': executeControl '" + c.executeControl
            + "' is unsupported; use timeStep or writeTime");
    if (c.writeControl != "timeStep" && c.writeControl != "writeTime" && c.writeControl != "none")
        throw std::runtime_error("yPlus '" + object + "': writeControl '" + c.writeControl
            + "' is unsupported; use timeStep, writeTime, or none");
    if (c.executeControl == "timeStep") YPlusCadence::checkInterval(object, "executeInterval", c.executeInterval);
    if (c.writeControl == "timeStep") YPlusCadence::checkInterval(object, "writeInterval", c.writeInterval);
    return c;
}

class YPlusWriter
{
public:
    YPlusWriter(const std::string& caseDir, const std::string& object, const std::string& startTime,
                scalar nu, const std::string& turbulence, const std::string& formula,
                const YPlusCadence& cadence)
      : root_((std::filesystem::path(caseDir) / "postProcessing" / "braeYPlus" / object / startTime).string()),
        metadata_(root_ + "/metadata.dat"), object_(object), cadence_(cadence)
    {
        const std::filesystem::path op(object);
        if (object.empty() || op.is_absolute() || op.has_parent_path() || object == "." || object == "..")
            throw std::runtime_error("yPlus '" + object + "': object name is not a safe single output-path component");
        std::error_code ec;
        std::filesystem::create_directories(std::filesystem::path(root_).parent_path(), ec);
        if (ec) throw std::runtime_error("yPlus: cannot create Brae output parent '"
            + std::filesystem::path(root_).parent_path().string() + "': " + ec.message());
        if (std::filesystem::exists(root_))
            throw std::runtime_error("yPlus '" + object + "': Brae output already exists; refusing collision: '" + root_ + "'");
        const bool created = std::filesystem::create_directory(root_, ec);
        if (ec) throw std::runtime_error("yPlus: cannot create Brae output '" + root_ + "': " + ec.message());
        if (!created)
            throw std::runtime_error("yPlus '" + object + "': Brae output appeared during creation; refusing collision: '" + root_ + "'");
        face_.open(root_ + "/faceValues.dat", std::ios::out | std::ios::app);
        summary_.open(root_ + "/patchSummary.dat", std::ios::out | std::ios::app);
        if (!face_ || !summary_) throw std::runtime_error("yPlus: cannot create Brae evidence files in '" + root_ + "'");
        face_ << "# Brae simpleFoam yPlus face evidence (Brae-owned; object=" << object << ")\n"
              << "# columns: solver_time iteration patch patch_local_face global_face face_area near_wall_distance yPlus\n";
        summary_ << "# Brae simpleFoam yPlus patch summaries (Brae-owned; object=" << object << ")\n"
                 << "# columns: solver_time iteration patch face_count total_area area_weighted_mean_yPlus minimum maximum percent_area_yPlus_30_300\n";
        face_ << std::scientific << std::setprecision(17);
        summary_ << std::scientific << std::setprecision(17);
        std::ofstream meta(metadata_, std::ios::out | std::ios::app);
        if (!meta) throw std::runtime_error("yPlus: cannot create metadata '" + metadata_ + "'");
        meta << "provenance=Brae-owned simpleFoam solver-owned yPlus; not OpenFOAM output\n"
             << "object=" << object << "\nsolver=simpleFoam\nturbulence_model=" << turbulence
             << "\nmolecular_nu=" << std::setprecision(17) << nu << "\nformula_path=" << formula << "\n"
             << "execute_control=" << cadence.executeControl << "\nexecute_interval=" << cadence.executeInterval
             << "\nwrite_control=" << cadence.writeControl << "\nwrite_interval=" << cadence.writeInterval << "\n"
             << "openfoam_yPlus_field_read=false\nwall_patches=geometric patch type wall only\n";
    }

    void sample(const YPlusSample& s)
    {
        for (const auto& v : s.faces)
            face_ << v.time << ' ' << v.iteration << ' ' << v.patch << ' ' << v.patchLocalFace << ' '
                  << v.globalFace << ' ' << v.area << ' ' << v.y << ' ' << v.yPlus << '\n';
        for (const auto& v : s.patches)
            summary_ << v.time << ' ' << v.iteration << ' ' << v.patch << ' ' << v.faceCount << ' '
                     << v.totalArea << ' ' << v.areaMean << ' ' << v.minimum << ' ' << v.maximum << ' '
                     << v.percentBand << '\n';
        face_.flush();
        summary_.flush();
        if (!face_ || !summary_) throw std::runtime_error("yPlus: failed while writing Brae evidence");
        ++samples_;
        lastIteration_ = s.faces.empty() ? lastIteration_ : s.faces.front().iteration;
    }

    void finish(const std::string& stoppingReason, int completedIteration)
    {
        if (completedIteration < 0) throw std::runtime_error("yPlus: completed iteration cannot be negative");
        std::ofstream meta(metadata_, std::ios::out | std::ios::app);
        if (!meta) throw std::runtime_error("yPlus: cannot append metadata '" + metadata_ + "'");
        meta << "stopping_reason=" << stoppingReason << "\ncompleted_iteration=" << completedIteration
             << "\nsample_count=" << samples_ << "\n";
        meta.flush();
        if (!meta) throw std::runtime_error("yPlus: failed while writing completion metadata");
        finalized_ = true;
    }

    ~YPlusWriter() noexcept
    {
        if (!finalized_)
        {
            try
            {
                std::ofstream meta(metadata_, std::ios::out | std::ios::app);
                if (meta) meta << "stopping_reason=aborted\ncompleted_iteration=" << lastIteration_
                               << "\nsample_count=" << samples_ << "\n";
            }
            catch (...) {}
        }
    }

    const std::string& root() const { return root_; }
    const std::string& metadataPath() const { return metadata_; }
    int samples() const { return samples_; }
    const YPlusCadence& cadence() const { return cadence_; }

private:
    std::string root_, metadata_, object_;
    YPlusCadence cadence_;
    std::ofstream face_, summary_;
    int samples_ = 0;
    int lastIteration_ = 0;
    bool finalized_ = false;
};

} // namespace brae
