// Job execution and GPU probing.
//
// The runner is tested against fake `brae` binaries written by the test itself -- one that succeeds, one that
// fails, one that hangs, one that is missing. Those are the four things that actually happen on a contributor's
// machine, and the last two are the ones that decide whether a node comes back or is lost until someone
// notices.
#include "gpu_probe.h"
#include "runner.h"

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

using namespace brae::node;

static int failures = 0;

static void check(bool cond, const std::string& what)
{
    if (cond) { std::printf("ok:   %s\n", what.c_str()); return; }
    std::printf("FAIL: %s\n", what.c_str());
    ++failures;
}

static void checkEq(const std::string& got, const std::string& want, const std::string& what)
{
    if (got == want) { std::printf("ok:   %s\n", what.c_str()); return; }
    std::printf("FAIL: %s\n        got:  %s\n        want: %s\n", what.c_str(), got.c_str(), want.c_str());
    ++failures;
}

static std::string gTmp;

static std::string writeScript(const std::string& name, const std::string& body)
{
    const std::string path = gTmp + "/" + name;
    std::ofstream f(path);
    f << "#!/bin/sh\n" << body;
    f.close();
    ::chmod(path.c_str(), 0755);
    return path;
}

static RunSettings settings(const std::string& binary)
{
    RunSettings s;
    s.braeBinary = binary;
    s.scratchRoot = gTmp + "/jobs";
    s.defaultTimeoutSeconds = 10;
    s.killGraceSeconds = 1;
    return s;
}

static JobRequest benchmarkJob()
{
    JobRequest j;
    j.jobId = "job-0182";
    j.type = JobType::BraeBenchmark;
    j.sample = "pimplefoam/pitzDaily-1M";
    return j;
}

// ---- progress parsing ---------------------------------------------------------------------------------------

static void testProgress()
{
    int p = -1;
    check(parseProgressLine("progress: 64", p) && p == 64, "progress: plain line");
    check(parseProgressLine("progress:7", p) && p == 7, "progress: no space");
    check(parseProgressLine("  progress:  100  ", p) && p == 100, "progress: surrounding whitespace");
    check(parseProgressLine("progress: 250", p) && p == 100, "progress: clamped to 100");
    check(!parseProgressLine("Time = 0.01", p), "progress: ordinary solver output is not progress");
    check(!parseProgressLine("progress: abc", p), "progress: non-numeric is refused");
    check(!parseProgressLine("progress:", p), "progress: empty value is refused");
    check(!parseProgressLine("", p), "progress: empty line");
    check(!parseProgressLine("progress: 99999999999999999999", p), "progress: absurd value refused, not wrapped");
}

// ---- scratch directory --------------------------------------------------------------------------------------

static void testScratchPath()
{
    RunSettings s = settings("/bin/true");
    std::string dir;
    check(jobScratchDir(s, "job-0182", dir), "scratch: accepts a normal job id");
    checkEq(dir, s.scratchRoot + "/job-0182", "scratch: path is under the root");

    // A job id arrives from the server and becomes a path. None of these may escape.
    for (const char* bad : {"../../etc", "a/b", "/abs", ".hidden", "-rf", "", "has space", "semi;colon"})
        check(!jobScratchDir(s, bad, dir), std::string("scratch: refuses job id ") + (*bad ? bad : "(empty)"));

    check(!jobScratchDir(s, std::string(65, 'a'), dir), "scratch: refuses an over-long job id");
}

// ---- running ---------------------------------------------------------------------------------------------

static void testSuccessfulRun()
{
    const std::string brae = writeScript("brae_ok",
        "echo 'progress: 10' >&2\n"
        "echo 'progress: 64' >&2\n"
        // argv is [brae, benchmark, <sample>], so inside the script the sample is $2.
        "printf '{\"sample\":\"%s\",\"runtime_s\":1.5,\"success\":true}' \"$2\" > brae-benchmark.json\n"
        "echo done\n");

    int lastProgress = -1;
    int calls = 0;
    const JobOutcome o = runJob(benchmarkJob(), settings(brae),
                                [&](int p) { lastProgress = p; ++calls; });

    check(o.success, "run: a clean run succeeds");
    checkEq(o.error, "", "run: no error on success");
    check(o.runtimeSeconds > 0, "run: records a runtime");
    check(calls == 2, "run: every progress line is reported");
    check(lastProgress == 64, "run: the last progress value is the latest");
    check(o.resultJson.find("pimplefoam/pitzDaily-1M") != std::string::npos,
          "run: reads brae-benchmark.json from the scratch dir, and the sample reached brae as argv[3]");
}

static void testFailures()
{
    {
        const std::string brae = writeScript("brae_fail", "echo 'brae ERROR: boom' >&2\nexit 1\n");
        const JobOutcome o = runJob(benchmarkJob(), settings(brae), nullptr);
        check(!o.success, "run: a non-zero exit fails");
        checkEq(o.error, "solver_exit_1", "run: the exit code is reported as a class");
    }
    {
        // Exit zero with no metrics file is not a success: the file IS the job.
        const std::string brae = writeScript("brae_empty", "echo nothing\n");
        const JobOutcome o = runJob(benchmarkJob(), settings(brae), nullptr);
        check(!o.success, "run: exit 0 with no result file is not a success");
        checkEq(o.error, "no_result_file", "run: says the metrics are missing");
    }
    {
        const JobOutcome o = runJob(benchmarkJob(), settings(gTmp + "/does_not_exist"), nullptr);
        check(!o.success, "run: a missing brae binary fails");
        checkEq(o.error, "brae_not_found", "run: names the missing binary case");
    }
    {
        // The case that matters most: a job that hangs must not hold the node forever.
        const std::string brae = writeScript("brae_hang", "sleep 300\n");
        RunSettings s = settings(brae);
        s.defaultTimeoutSeconds = 1;
        s.killGraceSeconds = 1;
        const auto begin = std::time(nullptr);
        const JobOutcome o = runJob(benchmarkJob(), s, nullptr);
        const auto took = std::time(nullptr) - begin;
        check(!o.success, "run: a hung job fails");
        checkEq(o.error, "timeout", "run: reported as a timeout");
        check(took < 20, "run: the deadline is actually enforced, not waited out");
    }
    {
        // A child that ignores SIGTERM still has to die.
        const std::string brae = writeScript("brae_stubborn", "trap '' TERM\nsleep 300\n");
        RunSettings s = settings(brae);
        s.defaultTimeoutSeconds = 1;
        s.killGraceSeconds = 1;
        const auto begin = std::time(nullptr);
        const JobOutcome o = runJob(benchmarkJob(), s, nullptr);
        check(!o.success, "run: a job ignoring SIGTERM still fails");
        check(std::time(nullptr) - begin < 25, "run: SIGKILL follows SIGTERM after the grace period");
    }
    {
        JobRequest j = benchmarkJob();
        j.sample = "; rm -rf /";
        const std::string brae = writeScript("brae_never", "touch SHOULD_NOT_RUN\n");
        const JobOutcome o = runJob(j, settings(brae), nullptr);
        check(!o.success, "run: a rejected sample never reaches a process");
        check(o.error.find("rejected") == 0, "run: says it was rejected before running");
        check(o.runtimeSeconds == 0, "run: nothing was executed at all");
    }
    {
        JobRequest j = benchmarkJob();
        j.type = JobType::Unknown;
        const JobOutcome o = runJob(j, settings("/bin/true"), nullptr);
        check(!o.success, "run: an unknown job type never reaches a process");
    }
}

static void testScratchIsCleaned()
{
    const std::string brae = writeScript("brae_litter",
        "mkdir -p sub/dir\n"
        "echo junk > sub/dir/file\n"
        "printf '{\"ok\":true}' > brae-benchmark.json\n");
    const RunSettings s = settings(brae);
    const JobOutcome o = runJob(benchmarkJob(), s, nullptr);
    check(o.success, "cleanup: the littering run succeeded");

    struct stat st {};
    check(::stat((s.scratchRoot + "/job-0182").c_str(), &st) != 0,
          "cleanup: the scratch directory is removed, including subdirectories");
}

// ---- GPU probe ----------------------------------------------------------------------------------------------

static void testGpuProbe()
{
    // The no-driver path: the agent must start, report nothing, and say why -- never crash.
    ::setenv("BRAE_NVML_LIB", "libnvidia-ml-does-not-exist.so.999", 1);
    const GpuProbeResult none = probeGpus();
    check(!none.nvmlAvailable, "gpu: reports NVML unavailable when the library is absent");
    check(none.gpus.empty(), "gpu: no GPUs without NVML");
    check(!none.unavailableReason.empty(), "gpu: says why NVML is unavailable");
    check(probeGpuState().empty(), "gpu: state probe is also empty and does not crash");
    ::unsetenv("BRAE_NVML_LIB");

    // The real path, when this machine has a driver. Skipped rather than failed elsewhere, so the suite still
    // runs on a CI box with no GPU.
    const GpuProbeResult real = probeGpus();
    if (!real.nvmlAvailable)
    {
        std::printf("skip: no NVML on this machine (%s)\n", real.unavailableReason.c_str());
        return;
    }
    check(!real.gpus.empty(), "gpu: finds at least one GPU on a machine with a driver");
    for (const GpuIdentity& g : real.gpus)
    {
        check(!g.model.empty(), "gpu: model is populated");
        // Must hold on unified-memory parts too. On a GB10 NVML answers NVML_ERROR_NOT_SUPPORTED with total=0,
        // and an earlier version dropped the card entirely -- so the machine reported no GPUs and could not
        // register. The API also requires vram_mb > 0, so a zero here is not merely cosmetic.
        check(g.vramMb > 0, "gpu: vram is positive even when NVML cannot report it (unified memory)");
        check(g.uuidHash.size() == 64, "gpu: uuid is reported as a 64-char hash");
        check(g.uuidHash.find("GPU-") == std::string::npos, "gpu: the RAW uuid never appears in the identity");
        check(!g.computeCapability.empty(), "gpu: compute capability is populated");
        std::printf("      detected: %s, %d MB, cc %s, driver %s\n",
                    g.model.c_str(), g.vramMb, g.computeCapability.c_str(), g.driverVersion.c_str());
    }
    for (const GpuState& s : probeGpuState())
    {
        check(s.memoryUsedMb <= s.memoryTotalMb, "gpu: used memory never exceeds total (the API rejects that)");
        check(s.utilizationPercent >= 0 && s.utilizationPercent <= 100, "gpu: utilization is a percentage");
    }
}

int main()
{
    char dir[] = "/tmp/brae-runner-testXXXXXX";
    if (!mkdtemp(dir)) { std::printf("FAIL: cannot create temp dir\n"); return 1; }
    gTmp = dir;

    testProgress();
    testScratchPath();
    testSuccessfulRun();
    testFailures();
    testScratchIsCleaned();
    testGpuProbe();

    if (failures == 0) std::printf("PASS: brae-agent runner and gpu probe\n");
    else std::printf("FAILED: %d check(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
