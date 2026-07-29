#pragma once
// The agent's HTTP transport, behind an interface.
//
// Two reasons it is an interface rather than direct libcurl calls. The protocol above it (api_client.h) is
// where the agent's actual behaviour lives -- what it sends, how it reads a reply, what it does when the reply
// is nonsense -- and that behaviour must be testable exhaustively, including the failures a real server will
// not produce on demand: a 500 mid-registration, a truncated body, a redirect to another host, a timeout. The
// second reason is that it keeps the libcurl-specific code down to one small adapter with no logic in it.
//
// Security posture, enforced in the curl backend and asserted in its tests:
//   * TLS verification is ALWAYS on. There is no flag to turn it off -- not even behind an environment
//     variable, because that flag is exactly what an attacker talks a user into setting.
//   * Redirects are NOT followed. A 302 would otherwise hand the node's bearer token to whatever host the
//     redirect names.
//   * Responses are size-capped and time-bounded, so a hung or hostile server cannot exhaust the node.
//   * The Authorization header is never logged.
#include <map>
#include <string>
#include <vector>

namespace brae::node {

struct HttpResponse
{
    long status = 0;            // 0 means the request never completed; see `error`
    std::string body;
    std::string error;          // transport-level failure, safe to log (carries no credentials)

    bool transportFailed() const { return status == 0; }
    bool ok() const { return status >= 200 && status < 300; }
};

struct HttpRequest
{
    std::string method = "GET";
    std::string url;
    std::string body;
    std::string bearer;         // sent as Authorization: Bearer <...>; never logged
    int timeoutSeconds = 15;
};

class HttpClient
{
public:
    virtual ~HttpClient() = default;
    virtual HttpResponse send(const HttpRequest& req) = 0;
};

// Cap on a response body. Every reply this agent understands is a few hundred bytes; anything approaching this
// is a server misbehaving, and reading it to the end would be the node's problem, not the server's.
constexpr std::size_t kMaxResponseBytes = 256u * 1024u;

// Join a base URL and a path without producing "//" or dropping a separator. Tiny, but it is the thing that
// silently produces 404s against a base URL written with a trailing slash.
std::string urlJoin(const std::string& base, const std::string& path);

}  // namespace brae::node
