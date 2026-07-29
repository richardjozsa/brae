#include "identity.h"

#include "json.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace brae::node {

std::string defaultIdentityPath()
{
    if (const char* over = std::getenv("BRAE_IDENTITY_PATH")) return over;
    return "/etc/brae/node.json";
}

IdentityResult loadIdentity(const std::string& path)
{
    IdentityResult r;

    struct stat st {};
    if (::stat(path.c_str(), &st) != 0)
    {
        r.error = "no identity at " + path + " (this machine is not registered)";
        return r;
    }

    // The token is only a secret while the file is. A permissive mode is a refusal, not a warning: continuing
    // would mean every local user can act as this node, and nobody would ever notice.
    if (st.st_mode & (S_IRWXG | S_IRWXO))
    {
        char mode[8];
        std::snprintf(mode, sizeof mode, "%04o", st.st_mode & 07777);
        r.error = std::string(path) + " is mode " + mode + "; it holds the node token and must be 0600. "
                  "Fix with: chmod 600 " + path;
        return r;
    }

    std::ifstream f(path);
    if (!f) { r.error = "cannot read " + path; return r; }
    const std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    Json j;
    std::string err;
    if (!Json::parse(text, j, err)) { r.error = path + " is not valid JSON: " + err; return r; }

    r.identity.nodeId = j["node_id"].asString();
    r.identity.nodeToken = j["node_token"].asString();
    r.identity.apiUrl = j["api_url"].asString();
    r.identity.registeredAt = j["registered_at"].asString();
    if (!r.identity.valid())
    {
        // Deliberately does not say which field, and never echoes a value: the missing one may be the token.
        r.error = path + " is missing node_id, node_token or api_url";
        return r;
    }
    r.ok = true;
    return r;
}

bool saveIdentity(const std::string& path, const Identity& id, std::string& error)
{
    error.clear();
    if (!id.valid()) { error = "refusing to write an incomplete identity"; return false; }

    const std::size_t slash = path.find_last_of('/');
    if (slash != std::string::npos)
    {
        const std::string dir = path.substr(0, slash);
        struct stat st {};
        if (::stat(dir.c_str(), &st) != 0 && ::mkdir(dir.c_str(), 0700) != 0)
        {
            error = "cannot create " + dir + ": " + std::strerror(errno);
            return false;
        }
    }

    Json j = Json::object();
    j.set("node_id", Json::str(id.nodeId));
    j.set("node_token", Json::str(id.nodeToken));
    j.set("api_url", Json::str(id.apiUrl));
    j.set("registered_at", Json::str(id.registeredAt));

    // Temp file at 0600, then rename: a reader either sees the old identity or the new one, never a partial
    // token. Created with open() rather than ofstream so the mode is right before any bytes exist.
    const std::string tmp = path + ".tmp";
    const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) { error = "cannot create " + tmp + ": " + std::strerror(errno); return false; }
    const std::string text = j.dump() + "\n";
    const ssize_t written = ::write(fd, text.data(), text.size());
    ::fsync(fd);
    ::close(fd);
    if (written != static_cast<ssize_t>(text.size()))
    {
        ::unlink(tmp.c_str());
        error = "short write to " + tmp;
        return false;
    }
    if (::rename(tmp.c_str(), path.c_str()) != 0)
    {
        ::unlink(tmp.c_str());
        error = "cannot install " + path + ": " + std::strerror(errno);
        return false;
    }
    ::chmod(path.c_str(), 0600);
    return true;
}

bool removeIdentity(const std::string& path, std::string& error)
{
    error.clear();
    if (::unlink(path.c_str()) != 0 && errno != ENOENT)
    {
        error = "cannot remove " + path + ": " + std::strerror(errno);
        return false;
    }
    return true;
}

}  // namespace brae::node
