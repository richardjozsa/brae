#include "runner.h"

#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fstream>
#include <sstream>
#include <vector>

#include <dirent.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

namespace brae::node {
namespace {

using Clock = std::chrono::steady_clock;

void setNonBlocking(int fd)
{
    const int flags = ::fcntl(fd, F_GETFL, 0);
    if (flags >= 0) ::fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

void removeTree(const std::string& path)
{
    // Small, explicit, and deliberately not a shell call. Depth is bounded by what brae writes into a case.
    DIR* d = ::opendir(path.c_str());
    if (!d) { ::unlink(path.c_str()); return; }
    while (struct dirent* e = ::readdir(d))
    {
        const std::string name = e->d_name;
        if (name == "." || name == "..") continue;
        const std::string child = path + "/" + name;
        struct stat st {};
        if (::lstat(child.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) removeTree(child);
        else ::unlink(child.c_str());
    }
    ::closedir(d);
    ::rmdir(path.c_str());
}

}  // namespace

bool parseProgressLine(const std::string& line, int& percent)
{
    const std::size_t at = line.find("progress:");
    if (at == std::string::npos) return false;
    std::size_t i = at + 9;
    while (i < line.size() && std::isspace(static_cast<unsigned char>(line[i]))) ++i;
    if (i >= line.size() || !std::isdigit(static_cast<unsigned char>(line[i]))) return false;
    int value = 0;
    while (i < line.size() && std::isdigit(static_cast<unsigned char>(line[i])))
    {
        value = value * 10 + (line[i] - '0');
        if (value > 100000) return false;                    // nonsense; do not wrap into something plausible
        ++i;
    }
    percent = value > 100 ? 100 : value;                     // clamp: progress is a percentage, whatever is printed
    return true;
}

bool jobScratchDir(const RunSettings& s, const std::string& jobId, std::string& out)
{
    // A job id comes from the server, and it is about to become a filesystem path. Restrict it to a shape that
    // cannot escape the scratch root -- no separators, no dots, no leading dash.
    if (jobId.empty() || jobId.size() > 64) return false;
    if (jobId.front() == '-' || jobId.front() == '.') return false;
    for (const char c : jobId)
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_')) return false;
    out = s.scratchRoot + "/" + jobId;
    return true;
}

JobOutcome runJob(const JobRequest& job, const RunSettings& settings, const ProgressFn& onProgress)
{
    JobOutcome outcome;
    outcome.jobId = job.jobId;

    const ArgvResult argv = argvFor(job, settings.braeBinary);
    if (!argv.ok)
    {
        outcome.error = "rejected: " + argv.reason;
        return outcome;
    }

    std::string dir;
    if (!jobScratchDir(settings, job.jobId, dir))
    {
        outcome.error = "rejected: unusable job id";
        return outcome;
    }

    removeTree(dir);
    {
        // Create the whole chain 0700. The scratch root may not exist on a fresh install.
        std::string built;
        std::stringstream ss(settings.scratchRoot);
        std::string part;
        const bool absolute = !settings.scratchRoot.empty() && settings.scratchRoot.front() == '/';
        while (std::getline(ss, part, '/'))
        {
            if (part.empty()) continue;
            built += (built.empty() && !absolute) ? part : "/" + part;
            ::mkdir(built.c_str(), 0700);
        }
        if (::mkdir(dir.c_str(), 0700) != 0 && errno != EEXIST)
        {
            outcome.error = std::string("cannot create scratch dir: ") + std::strerror(errno);
            return outcome;
        }
    }

    int outPipe[2] = {-1, -1};
    int errPipe[2] = {-1, -1};
    if (::pipe(outPipe) != 0 || ::pipe(errPipe) != 0)
    {
        outcome.error = "cannot create pipes";
        removeTree(dir);
        return outcome;
    }

    const pid_t pid = ::fork();
    if (pid < 0)
    {
        outcome.error = "cannot fork";
        removeTree(dir);
        return outcome;
    }
    if (pid == 0)
    {
        // Child. Explicit argv, no shell anywhere. cwd is the scratch dir so brae-benchmark.json lands there.
        ::close(outPipe[0]);
        ::close(errPipe[0]);
        ::dup2(outPipe[1], STDOUT_FILENO);
        ::dup2(errPipe[1], STDERR_FILENO);
        ::close(outPipe[1]);
        ::close(errPipe[1]);
        if (::chdir(dir.c_str()) != 0) ::_exit(126);
        ::setpgid(0, 0);                       // own process group, so a timeout kills the whole tree

        std::vector<char*> av;
        av.reserve(argv.argv.size() + 1);
        for (const std::string& a : argv.argv) av.push_back(const_cast<char*>(a.c_str()));
        av.push_back(nullptr);
        ::execv(av[0], av.data());
        ::_exit(127);                          // exec failed: brae is not where we were told it is
    }

    ::close(outPipe[1]);
    ::close(errPipe[1]);
    setNonBlocking(outPipe[0]);
    setNonBlocking(errPipe[0]);

    const int timeout = job.timeoutSeconds > 0 ? job.timeoutSeconds : settings.defaultTimeoutSeconds;
    const auto started = Clock::now();
    std::string stdoutText, stderrPending;
    bool termSent = false;
    auto termAt = Clock::now();
    int status = 0;
    bool reaped = false;

    while (!reaped)
    {
        char buf[4096];
        ssize_t n;
        while ((n = ::read(outPipe[0], buf, sizeof buf)) > 0)
            if (stdoutText.size() < (1u << 20)) stdoutText.append(buf, static_cast<std::size_t>(n));
        while ((n = ::read(errPipe[0], buf, sizeof buf)) > 0)
        {
            stderrPending.append(buf, static_cast<std::size_t>(n));
            std::size_t nl;
            while ((nl = stderrPending.find('\n')) != std::string::npos)
            {
                const std::string line = stderrPending.substr(0, nl);
                stderrPending.erase(0, nl + 1);
                int percent = 0;
                if (parseProgressLine(line, percent) && onProgress) onProgress(percent);
            }
            if (stderrPending.size() > (1u << 16)) stderrPending.clear();   // a line that never ends
        }

        const pid_t got = ::waitpid(pid, &status, WNOHANG);
        if (got == pid) { reaped = true; break; }

        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(Clock::now() - started).count();
        if (!termSent && elapsed >= timeout)
        {
            // Deadline. Ask the process group to stop, then insist.
            ::kill(-pid, SIGTERM);
            termSent = true;
            termAt = Clock::now();
            outcome.error = "timeout";
        }
        if (termSent
            && std::chrono::duration_cast<std::chrono::seconds>(Clock::now() - termAt).count()
               >= settings.killGraceSeconds)
        {
            ::kill(-pid, SIGKILL);
        }
        ::usleep(50 * 1000);
    }

    ::close(outPipe[0]);
    ::close(errPipe[0]);
    outcome.runtimeSeconds =
        std::chrono::duration<double>(Clock::now() - started).count();

    // brae writes its metrics to brae-benchmark.json in the working directory, always, with no flag asked for.
    {
        std::ifstream f(dir + "/brae-benchmark.json");
        if (f)
        {
            std::stringstream ss;
            ss << f.rdbuf();
            outcome.resultJson = ss.str();
        }
    }

    if (outcome.error == "timeout")
    {
        outcome.success = false;
    }
    else if (WIFEXITED(status) && WEXITSTATUS(status) == 0)
    {
        if (outcome.resultJson.empty())
        {
            // Exit zero with no metrics is not a success we can report: the whole point of the job is the file.
            outcome.success = false;
            outcome.error = "no_result_file";
        }
        else outcome.success = true;
    }
    else if (WIFEXITED(status))
    {
        const int code = WEXITSTATUS(status);
        outcome.success = false;
        outcome.error = code == 127 ? "brae_not_found"
                      : code == 126 ? "scratch_dir_unusable"
                                    : "solver_exit_" + std::to_string(code);
    }
    else if (WIFSIGNALED(status))
    {
        outcome.success = false;
        outcome.error = "killed_signal_" + std::to_string(WTERMSIG(status));
    }
    else
    {
        outcome.success = false;
        outcome.error = "unknown_failure";
    }

    removeTree(dir);
    return outcome;
}

}  // namespace brae::node
