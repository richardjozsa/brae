#pragma once
// Brae-owned, per-iteration force-coefficient history for the steady simpleFoam forceCoeffs object.
#include "forces.cuh"
#include "write_control.cuh"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>

namespace brae {

class ForceHistoryWriter
{
public:
    ForceHistoryWriter(const std::string& caseDir, const ForceCoeffsConfig& c, scalar startTimeVal)
      : path_((std::filesystem::path(caseDir) / "postProcessing" / "braeForceHistory" / c.name
               / WriteControl::timeName(startTimeVal) / "coefficient.dat").string()),
        metadataPath_(path_ + ".meta"), object_(c.name)
    {
        const std::filesystem::path objectPath(c.name);
        if (c.name.empty() || objectPath.is_absolute() || objectPath.has_parent_path()
            || c.name == "." || c.name == "..")
            throw std::runtime_error("forceCoeffs '" + c.name
                                     + "': object name is not a safe single output-path component");

        std::error_code ec;
        std::filesystem::create_directories(std::filesystem::path(path_).parent_path(), ec);
        if (ec)
            throw std::runtime_error("forceCoeffs: cannot create Brae output directory '"
                                     + std::filesystem::path(path_).parent_path().string() + "': " + ec.message());
        if (std::filesystem::exists(path_))
            throw std::runtime_error("forceCoeffs '" + c.name + "': Brae history output already exists; refusing "
                                     "to overwrite a file not written by this run: '" + path_ + "'");
        if (std::filesystem::exists(metadataPath_))
            throw std::runtime_error("forceCoeffs '" + c.name + "': Brae history metadata already exists; refusing "
                                     "to append to a file not written by this run: '" + metadataPath_ + "'");

        // The existence check is deliberate and this is append-only: no ios::trunc and no path owned by
        // OpenFOAM. A stale Brae file therefore stops the run instead of destroying prior evidence.
        out_.open(path_, std::ios::out | std::ios::app);
        if (!out_)
            throw std::runtime_error("forceCoeffs: cannot create Brae history output '" + path_ + "'");

        out_ << "# Brae simpleFoam forceCoeffs history (Brae-owned; object=" << c.name << ")\n";
        out_ << "# output: " << path_ << "\n";
        out_ << "# startTime: " << std::setprecision(17) << startTimeVal << "\n";
        out_ << "# normalization: rho=" << c.rhoName << " rhoInf=" << c.rhoInf
             << " magUInf=" << c.magUInf << " Aref=" << c.Aref << " lRef=" << c.lRef
             << " pRef=" << c.pRef << " dragDir=(" << c.dragDir.x << ' ' << c.dragDir.y << ' ' << c.dragDir.z
             << ") liftDir=(" << c.liftDir.x << ' ' << c.liftDir.y << ' ' << c.liftDir.z
             << ") pitchAxis=(" << c.pitchAxis.x << ' ' << c.pitchAxis.y << ' ' << c.pitchAxis.z
             << ") CofR=(" << c.CofR.x << ' ' << c.CofR.y << ' ' << c.CofR.z << ") patches=(";
        for (std::size_t i = 0; i < c.patches.size(); ++i)
            out_ << (i ? " " : "") << c.patches[i];
        out_ << ")\n";
        out_ << "# columns: Time Cd Cl Cm Fx Fy Fz Fpx Fpy Fpz Fvx Fvy Fvz wallTime Iteration\n";
        out_ << "# column contract: Time=resolved OpenFOAM time; Cd/Cl/Cm=normalized total force and moment; "
                "Fx/Fy/Fz=total force; Fpx/Fpy/Fpz=pressure force; Fvx/Fvy/Fvz=viscous force; "
                "wallTime=seconds since solver start; Iteration=completed SIMPLE iteration\n";
        out_ << "# repeatability: GPU wall-face reduction uses atomicAdd and is not bitwise reproducible; "
                "repeat comparisons use abs tolerance 5e-6*max(1,|Cd|,|Cl|,|Cm|)\n";
        out_ << "# completion metadata: " << metadataPath_ << "\n";
        out_ << std::scientific << std::setprecision(17);
        out_.flush();
        if (!out_)
            throw std::runtime_error("forceCoeffs: failed while writing Brae history header '" + path_ + "'");
    }

    ~ForceHistoryWriter() noexcept
    {
        if (!finalized_)
        {
            try { appendMetadata("aborted", samples_); }
            catch (...) {}
        }
    }

    void sample(scalar time, int iteration, scalar wallTime, const ForceResult& f, const ForceCoeffs& c)
    {
        const vector total = f.total();
        out_ << time << ' ' << c.Cd << ' ' << c.Cl << ' ' << c.Cm << ' '
             << total.x << ' ' << total.y << ' ' << total.z << ' '
             << f.pressure.x << ' ' << f.pressure.y << ' ' << f.pressure.z << ' '
             << f.viscous.x << ' ' << f.viscous.y << ' ' << f.viscous.z << ' '
             << wallTime << ' ' << iteration << '\n';
        out_.flush();
        if (!out_) throw std::runtime_error("forceCoeffs: failed while writing Brae history '" + path_ + "'");
        ++samples_;
        last_ = c;
    }

    // Record a terminal state separately so teardown never has to rewrite the append-only history file.
    // Normal completion requires one sample per completed SIMPLE iteration. A numerical failure may stop after
    // a solver step that could not produce a finite sample, so its sample count is allowed to be smaller.
    void finish(const std::string& stoppingReason, int completedIterations)
    {
        if (completedIterations < 0)
            throw std::runtime_error("forceCoeffs: completed iteration count cannot be negative");
        if ((stoppingReason == "converged" || stoppingReason == "iteration_limit")
            && samples_ != completedIterations)
            throw std::runtime_error("forceCoeffs '" + object_ + "': " + stoppingReason
                                     + " requires one history sample per completed iteration, but recorded "
                                     + std::to_string(samples_) + " for " + std::to_string(completedIterations));
        appendMetadata(stoppingReason, samples_, completedIterations);
        finalized_ = true;
    }

    void recordOracleDiscrepancy(const ForceCoeffs& history, const ForceCoeffs& legacy, scalar tolerance)
    {
        std::ofstream meta(metadataPath_, std::ios::out | std::ios::app);
        if (!meta)
            throw std::runtime_error("forceCoeffs: cannot append discrepancy record '" + metadataPath_ + "'");
        meta << std::setprecision(17)
             << "oracle_discrepancy=history_vs_legacy"
             << " history_Cd=" << history.Cd << " legacy_Cd=" << legacy.Cd
             << " history_Cl=" << history.Cl << " legacy_Cl=" << legacy.Cl
             << " history_Cm=" << history.Cm << " legacy_Cm=" << legacy.Cm
             << " tolerance=" << tolerance << '\n';
        meta.flush();
        if (!meta)
            throw std::runtime_error("forceCoeffs: failed while writing discrepancy record '" + metadataPath_ + "'");
    }

    const std::string& path() const { return path_; }
    const std::string& metadataPath() const { return metadataPath_; }
    int samples() const { return samples_; }
    const ForceCoeffs& last() const { return last_; }

private:
    void appendMetadata(const std::string& stoppingReason, int sampleCount, int completedIterations = -1)
    {
        std::ofstream meta(metadataPath_, std::ios::out | std::ios::app);
        if (!meta)
            throw std::runtime_error("forceCoeffs: cannot write metadata '" + metadataPath_ + "'");
        meta << "stopping_reason=" << stoppingReason << '\n'
             << "sample_count=" << sampleCount << '\n';
        if (completedIterations >= 0)
            meta << "completed_iterations=" << completedIterations << '\n';
        meta.flush();
        if (!meta)
            throw std::runtime_error("forceCoeffs: failed while writing metadata '" + metadataPath_ + "'");
    }

    std::string path_;
    std::string metadataPath_;
    std::string object_;
    std::ofstream out_;
    int samples_ = 0;
    ForceCoeffs last_{};
    bool finalized_ = false;
};

} // namespace brae
