#pragma once
// brae -cases c1 c2 ... : run several cases at once, one per GPU (a mesh/parameter study), embedded in
// the brae binary. The parent process is a pure orchestrator (no CUDA): it forks one child `brae -case cX`
// per GPU, each pinned with CUDA_VISIBLE_DEVICES, tags the children's OpenFOAM-style residual output
// [GPUn case], and prints a summary. Cases beyond the GPU count queue. Because the parent never touches
// CUDA and the children exec a fresh image, there is no fork-after-CUDA-init hazard. GPU count comes from
// `nvidia-smi -L` (override with BRAE_GPUS); concurrency defaults to the GPU count (override with BRAE_JOBS).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <unistd.h>
#include <sys/wait.h>

namespace braesweep {

struct CaseState
{
    std::string name, path;
    int gpu = -1, iter = 0;
    std::string pResid = "-", uxResid = "-", status = "queued";
    double t0 = 0, t1 = 0;
};

inline std::string firstTok(const char* p)   // number up to the next comma/space
{
    std::string s;
    while (*p && *p != ',' && *p != ' ' && *p != '\n')
        s += *p++;
    return s.empty() ? std::string("-") : s;
}

inline double now()
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline int detectGPUs()
{
    if (const char* e = std::getenv("BRAE_GPUS"))
    {
        int n = std::atoi(e);
        if (n > 0) return n;
    }
    int n = 0;
    if (FILE* f = popen("nvidia-smi -L 2>/dev/null", "r"))
    {
        char b[256];
        while (fgets(b, sizeof b, f))
            if (b[0] != '\n') n++;
        pclose(f);
    }
    return n > 0 ? n : 1;
}

inline std::string selfExe()
{
    char b[4096];
    ssize_t k = readlink("/proc/self/exe", b, sizeof b - 1);
    if (k > 0)
    {
        b[k] = 0;
        return b;
    }
    return "brae";
}

inline std::string cellStr(const std::string& c)
{
    FILE* f = std::fopen((c + "/constant/polyMesh/owner").c_str(), "rb");
    if (!f) return "?";
    char b[2048];
    size_t n = std::fread(b, 1, sizeof b - 1, f);
    b[n] = 0;
    std::fclose(f);
    const char* p = std::strstr(b, "nCells:");
    if (!p) return "?";
    long nc = std::atol(p + 7);
    char o[32];
    if (nc >= 100000)
        std::snprintf(o, sizeof o, "%.2fM", nc / 1e6);
    else
        std::snprintf(o, sizeof o, "%ldk", nc / 1000);
    return o;
}

// Run one case as a child brae process pinned to `gpu`; tag its stdout and parse residuals.
inline void runOne(
    CaseState& st,
    int gpu,
    const std::string& self,
    std::mutex& outmx)
{
    st.gpu = gpu;
    st.status = "running";
    st.t0 = now();
    int fds[2];
    if (pipe(fds) != 0)
    {
        st.status = "failed";
        st.t1 = now();
        return;
    }
    pid_t pid = fork();
    if (pid == 0)   // child
    {
        close(fds[0]);
        dup2(fds[1], STDOUT_FILENO);
        dup2(fds[1], STDERR_FILENO);
        close(fds[1]);
        char g[16];
        std::snprintf(g, sizeof g, "%d", gpu);
        setenv("CUDA_VISIBLE_DEVICES", g, 1);
        execl(self.c_str(), "brae", "-case", st.path.c_str(), (char*)nullptr);
        _exit(127);
    }
    close(fds[1]);
    FILE* f = fdopen(fds[0], "r");
    char tag[288];
    std::snprintf(tag, sizeof tag, "[GPU%d %s]", gpu, st.name.c_str());
    char buf[2048];
    while (f && fgets(buf, sizeof buf, f))
    {
        size_t L = std::strlen(buf);
        if (L && buf[L-1] == '\n') buf[L-1] = 0;
        if (buf[0])
        {
            std::lock_guard<std::mutex> lk(outmx);
            std::printf("%s  %s\n", tag, buf);
            std::fflush(stdout);
        }
        if (std::strncmp(buf, "Time = ", 7) == 0) st.iter = std::atoi(buf + 7);
        if (const char* p = std::strstr(buf, "Solving for Ux, Initial residual = ")) st.uxResid = firstTok(p + 35);
        if (const char* p = std::strstr(buf, "Solving for p, Initial residual = "))  st.pResid  = firstTok(p + 34);
        if (std::strstr(buf, "-nan") || std::strstr(buf, "FATAL")) st.status = "diverged";
    }
    if (f) std::fclose(f);
    int ws = 0;
    waitpid(pid, &ws, 0);
    st.t1 = now();
    if (st.status == "running")
        st.status = (WIFEXITED(ws) && WEXITSTATUS(ws) == 0) ? "done" : "failed";
}

inline int runSweepCases(const std::vector<std::string>& paths)
{
    std::vector<CaseState> cases;
    for (auto c : paths)
    {
        while (c.size() > 1 && c.back() == '/')
            c.pop_back();
        CaseState s;
        s.path = c;
        s.name = c.substr(c.find_last_of('/') + 1);
        cases.push_back(std::move(s));
    }
    const int nGPU = detectGPUs();
    int jobs = nGPU;
    if (const char* e = std::getenv("BRAE_JOBS"))
    {
        int j = std::atoi(e);
        if (j > 0) jobs = j;
    }
    const std::string self = selfExe();
    std::printf("brae mesh study: %zu case(s), %d GPU(s) found, running %d at a time\n",
                cases.size(), nGPU, jobs);
    if ((int)cases.size() > nGPU && jobs <= nGPU)
        std::printf("  WARNING: %zu cases but only %d GPU(s); %d run at once, the other %zu queue. "
                    "add %zu more GPU(s) to run them all in parallel.\n",
                    cases.size(), nGPU, nGPU, cases.size() - nGPU, cases.size() - nGPU);
    if (jobs > nGPU)
        std::printf("  WARNING: BRAE_JOBS %d exceeds the %d GPU(s) found; cases will share GPUs and may run "
                    "out of GPU memory.\n", jobs, nGPU);
    std::fflush(stdout);

    std::atomic<size_t> next{0};
    std::mutex outmx;
    auto worker = [&](int wid)
    {
        const int gpu = wid % nGPU;
        for (;;)
        {
            size_t i = next.fetch_add(1);
            if (i >= cases.size()) break;
            runOne(cases[i], gpu, self, outmx);
        }
    };
    std::vector<std::thread> th;
    for (int w = 0; w < jobs; ++w)
        th.emplace_back(worker, w);
    for (auto& t : th)
        t.join();

    std::printf("\n%s\n", std::string(84, '=').c_str());
    std::printf("mesh study complete (%zu cases)\n\n", cases.size());
    std::printf("  %-16s %4s %9s %7s %11s %11s %9s   %s\n",
                "case", "GPU", "cells", "iters", "p resid", "Ux resid", "wall", "status");
    std::printf("  %s\n", std::string(78, '-').c_str());
    for (auto& s : cases)
    {
        int wall = (int)(s.t1 - s.t0);
        char w[16];
        std::snprintf(w, sizeof w, "%dm%02ds", wall / 60, wall % 60);
        std::printf("  %-16s %4d %9s %7d %11s %11s %9s   %s\n",
                    s.name.c_str(), s.gpu, cellStr(s.path).c_str(), s.iter,
                    s.pResid.c_str(), s.uxResid.c_str(), w, s.status.c_str());
    }
    std::printf("  full fields + time directories written to each case folder.\n");
    std::fflush(stdout);
    return 0;
}

} // namespace braesweep
