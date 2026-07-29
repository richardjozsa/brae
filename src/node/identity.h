#pragma once
// The node's permanent identity: /etc/brae/node.json.
//
// It holds the node token, which is the machine's whole authority on the network, so this file is stricter
// than a config reader normally would be:
//
//   * written 0600 and verified 0600 on read -- a token readable by every local user is not a secret;
//   * never logged, never put in a snapshot, never included in an error message (docs/10-security.md, rule 5);
//   * a machine keeps this across reboots and reinstalls of the service, which is what lets an archived node
//     come back with its history intact rather than as a new node.
#include <string>

namespace brae::node {

struct Identity
{
    std::string nodeId;
    std::string nodeToken;      // secret
    std::string apiUrl;
    std::string registeredAt;

    bool valid() const { return !nodeId.empty() && !nodeToken.empty() && !apiUrl.empty(); }
};

struct IdentityResult
{
    bool ok = false;
    std::string error;          // safe to log: never contains the token
    Identity identity;
};

// Default location. Overridable for tests and for a non-root dev run.
std::string defaultIdentityPath();

// Absent file -> ok=false with error "no identity", which is how `brae node status` reports "not registered"
// rather than as a failure.
IdentityResult loadIdentity(const std::string& path);

// Writes atomically (temp file + rename) at 0600. A half-written identity is worse than none: the agent would
// read a truncated token and authenticate as nobody, forever.
bool saveIdentity(const std::string& path, const Identity& id, std::string& error);

bool removeIdentity(const std::string& path, std::string& error);

}  // namespace brae::node
