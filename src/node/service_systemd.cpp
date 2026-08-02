// The systemd implementation of ServiceManager.
//
// Kept out of cli.cpp so the CLI logic tests link against a fake and never touch systemd. Everything here
// shells out to systemctl -- via fork/exec with an explicit argv, never a shell string.
//
// The unit is written with the hardening from docs/04-agent.md. It is not decoration: the agent runs work
// requested by a remote service, so it gets no more of the machine than it needs.
#include "cli.h"

#include <cerrno>
#include <cstring>
#include <fstream>
#include <sstream>

#include <fcntl.h>
#include <pwd.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

namespace brae::node {
namespace {

// The unprivileged account the unit runs as. Declared in the unit, created by ensureUser().
constexpr const char* kServiceUser = "brae";

int runQuiet(const std::vector<std::string>& argv, std::string* captured = nullptr)
{
    int pipefd[2] = {-1, -1};
    if (captured && ::pipe(pipefd) != 0) return -1;

    const pid_t pid = ::fork();
    if (pid < 0) return -1;
    if (pid == 0)
    {
        if (captured) { ::close(pipefd[0]); ::dup2(pipefd[1], STDOUT_FILENO); ::close(pipefd[1]); }
        else
        {
            const int devnull = ::open("/dev/null", O_WRONLY);
            if (devnull >= 0) { ::dup2(devnull, STDOUT_FILENO); ::close(devnull); }
        }
        const int devnull = ::open("/dev/null", O_WRONLY);
        if (devnull >= 0) { ::dup2(devnull, STDERR_FILENO); ::close(devnull); }

        std::vector<char*> av;
        for (const std::string& a : argv) av.push_back(const_cast<char*>(a.c_str()));
        av.push_back(nullptr);
        ::execvp(av[0], av.data());       // explicit argv; no shell anywhere
        ::_exit(127);
    }
    if (captured)
    {
        ::close(pipefd[1]);
        char buf[4096];
        ssize_t n;
        while ((n = ::read(pipefd[0], buf, sizeof buf)) > 0) captured->append(buf, static_cast<std::size_t>(n));
        ::close(pipefd[0]);
    }
    int status = 0;
    ::waitpid(pid, &status, 0);
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

class SystemdService : public ServiceManager
{
public:
    SystemdService(std::string unitPath, std::string execPath)
        : unitPath_(std::move(unitPath)), execPath_(std::move(execPath)) {}

    bool ensureUser(std::string& error) override
    {
        if (::getpwnam(kServiceUser) != nullptr) return true;      // already there; nothing to do

        // A system account: no home, no login shell, no password. It exists to own a process and a state
        // directory, nothing else. `video` and `render` are what NVML needs to see the GPU.
        const int rc = runQuiet({"useradd", "--system", "--no-create-home",
                                 "--shell", "/usr/sbin/nologin",
                                 "--groups", "video,render",
                                 "--comment", "Brae node agent", kServiceUser});
        if (rc != 0 && ::getpwnam(kServiceUser) == nullptr)
        {
            error = std::string("cannot create the '") + kServiceUser + "' system user (useradd exited "
                    + std::to_string(rc) + "); run with sudo";
            return false;
        }
        return true;
    }

    bool takeOwnership(const std::string& path, std::string& error) override
    {
        const struct passwd* pw = ::getpwnam(kServiceUser);
        if (pw == nullptr)
        {
            error = std::string("the '") + kServiceUser + "' user does not exist";
            return false;
        }
        // The parent directory too: the agent has to traverse /etc/brae to reach the identity inside it.
        const std::size_t slash = path.find_last_of('/');
        if (slash != std::string::npos)
            ::chown(path.substr(0, slash).c_str(), pw->pw_uid, pw->pw_gid);
        if (::chown(path.c_str(), pw->pw_uid, pw->pw_gid) != 0)
        {
            error = "cannot give " + path + " to " + kServiceUser + ": " + std::strerror(errno);
            return false;
        }
        ::chmod(path.c_str(), 0600);            // still readable only by its owner -- now the right owner
        return true;
    }

    bool install(std::string& error) override
    {
        std::ofstream f(unitPath_);
        if (!f)
        {
            error = "cannot write " + unitPath_ + " (" + std::strerror(errno) + "); run with sudo";
            return false;
        }
        f << unitText();
        f.close();
        ::chmod(unitPath_.c_str(), 0644);

        if (runQuiet({"systemctl", "daemon-reload"}) != 0)
        {
            error = "systemctl daemon-reload failed";
            return false;
        }
        if (runQuiet({"systemctl", "enable", "brae-agent"}) != 0)
        {
            error = "systemctl enable brae-agent failed";
            return false;
        }
        return true;
    }

    bool start(std::string& error) override
    {
        if (runQuiet({"systemctl", "restart", "brae-agent"}) != 0)
        {
            error = "systemctl restart brae-agent failed; see: journalctl -u brae-agent -n 50";
            return false;
        }
        return true;
    }

    bool stopAndRemove(std::string& error) override
    {
        runQuiet({"systemctl", "stop", "brae-agent"});
        runQuiet({"systemctl", "disable", "brae-agent"});
        if (::unlink(unitPath_.c_str()) != 0 && errno != ENOENT)
            error = "could not remove " + unitPath_;
        runQuiet({"systemctl", "daemon-reload"});
        return true;
    }

    std::string statusLine() override
    {
        std::string out;
        runQuiet({"systemctl", "is-active", "brae-agent"}, &out);
        while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) out.pop_back();
        return out.empty() ? "unknown" : out;
    }

private:
    std::string unitText() const
    {
        std::ostringstream u;
        u << "[Unit]\n"
             "Description=Brae node agent\n"
             "Documentation=https://github.com/simd-ai/brae\n"
             "After=network-online.target\n"
             "Wants=network-online.target\n"
             "\n"
             "[Service]\n"
             "Type=simple\n"
             "User=brae\n"
             "Group=brae\n"
             "ExecStart=" << execPath_ << " run\n"
             "Restart=always\n"
             "RestartSec=5\n"
             "\n"
             "# The agent runs work requested by a remote service. It gets no more of the machine than it needs.\n"
             "NoNewPrivileges=true\n"
             "PrivateTmp=true\n"
             "ProtectSystem=strict\n"
             "ProtectHome=true\n"
             "ProtectKernelTunables=true\n"
             "ProtectControlGroups=true\n"
             "RestrictSUIDSGID=true\n"
             "LockPersonality=true\n"
             "MemoryDenyWriteExecute=false\n"   // NVRTC compiles device code at run time
             "ReadWritePaths=/var/lib/brae\n"
             "StateDirectory=brae\n"
             "SupplementaryGroups=video render\n"
             "\n"
             "[Install]\n"
             "WantedBy=multi-user.target\n";
        return u.str();
    }

    std::string unitPath_;
    std::string execPath_;
};

class NoopService : public ServiceManager
{
public:
    // --no-service: a developer running the agent in the foreground as themselves. Creating a system account
    // and reassigning their identity file would be a surprise, so both are no-ops here.
    bool ensureUser(std::string&) override { return true; }
    bool takeOwnership(const std::string&, std::string&) override { return true; }
    bool install(std::string&) override { return true; }
    bool start(std::string&) override { return true; }
    bool stopAndRemove(std::string&) override { return true; }
    std::string statusLine() override { return "not managed (--no-service)"; }
};

}  // namespace

std::unique_ptr<ServiceManager> makeSystemdService(const std::string& unitPath, const std::string& execPath)
{
    return std::make_unique<SystemdService>(unitPath, execPath);
}

std::string systemInstallDir()
{
    return "/usr/local/bin";
}

bool isServiceReachablePath(const std::string& path)
{
    // A prefix test rather than an access() probe, because access() answers for whoever is asking -- and the
    // caller is root, for whom everything is reachable. The question is whether the *service* can reach it
    // after ProtectHome=true and ProtectSystem=strict, and that is decided by where the path is, not by us.
    if (path.empty() || path.front() != '/') return false;

    auto under = [&path](const char* prefix) {
        const std::string p(prefix);
        return path.size() > p.size() && path.compare(0, p.size(), p) == 0;
    };

    // ProtectHome=true hides these outright.
    if (under("/home/") || under("/root/") || under("/Users/")) return false;

    // Everything the service can still see and execute. /tmp is excluded on purpose: PrivateTmp=true gives the
    // unit its own empty /tmp, so a binary there would vanish from under it.
    return under("/usr/") || under("/opt/") || under("/srv/") || under("/var/lib/brae/");
}

std::unique_ptr<ServiceManager> makeNoopService()
{
    return std::make_unique<NoopService>();
}

}  // namespace brae::node
