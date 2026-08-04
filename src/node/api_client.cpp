#include "api_client.h"

#include "backoff.h"
#include "json.h"

namespace brae::node {

std::string urlJoin(const std::string& base, const std::string& path)
{
    if (base.empty()) return path;
    const bool baseSlash = base.back() == '/';
    const bool pathSlash = !path.empty() && path.front() == '/';
    if (baseSlash && pathSlash) return base + path.substr(1);
    if (!baseSlash && !pathSlash) return base + "/" + path;
    return base + path;
}

// ---- request bodies -------------------------------------------------------------------------------------------

std::string buildRegisterBody(const RegisterRequest& r)
{
    Json j = Json::object();
    j.set("display_name", Json::str(r.displayName));
    j.set("operating_system", Json::str(r.operatingSystem));
    j.set("architecture", Json::str(r.architecture));
    j.set("agent_version", Json::str(r.agentVersion));
    j.set("brae_version", Json::str(r.braeVersion));
    // Omitted rather than sent as 0 when unknown: the server stores null, and "unknown" and "zero watts" are
    // different facts. An older server that does not know the field ignores it either way.
    if (r.systemRamMb > 0) j.set("system_ram_mb", Json::num(r.systemRamMb));
    if (!r.timezone.empty()) j.set("timezone", Json::str(r.timezone));

    Json gpus = Json::array();
    for (const GpuIdentity& g : r.gpus)
    {
        Json e = Json::object();
        e.set("index", Json::num(g.index));
        e.set("model", Json::str(g.model));
        e.set("vram_mb", Json::num(g.vramMb));
        e.set("compute_capability", Json::str(g.computeCapability));
        e.set("driver_version", Json::str(g.driverVersion));
        e.set("uuid_hash", Json::str(g.uuidHash));      // never the raw UUID; see gpu_probe
        if (g.powerLimitW > 0) e.set("power_limit_w", Json::num(g.powerLimitW));
        if (g.cudaCores > 0) e.set("cuda_cores", Json::num(g.cudaCores));
        gpus.push(std::move(e));
    }
    j.set("gpus", std::move(gpus));
    return j.dump();
}

std::string buildSnapshotBody(const SnapshotRequest& s)
{
    Json j = Json::object();
    j.set("timestamp", Json::str(s.timestamp));
    j.set("status", Json::str(s.status));
    j.set("accepts_jobs", Json::boolean(s.acceptsJobs));
    j.set("agent_version", Json::str(s.agentVersion));

    Json gpus = Json::array();
    for (const GpuState& g : s.gpus)
    {
        Json e = Json::object();
        e.set("index", Json::num(g.index));
        e.set("utilization_percent", Json::num(g.utilizationPercent));
        e.set("memory_used_mb", Json::num(g.memoryUsedMb));
        e.set("memory_total_mb", Json::num(g.memoryTotalMb));
        e.set("temperature_celsius", Json::num(g.temperatureCelsius));
        gpus.push(std::move(e));
    }
    j.set("gpus", std::move(gpus));

    if (s.hasJob)
    {
        Json job = Json::object();
        job.set("job_id", Json::str(s.jobId));
        job.set("type", Json::str(jobTypeName(s.jobType)));
        job.set("progress_percent", Json::num(s.jobProgressPercent));
        j.set("job", std::move(job));
    }
    else
    {
        j.set("job", Json());
    }
    return j.dump();
}

std::string buildResultBody(const JobOutcome& o, const std::string& startedAt, const std::string& finishedAt)
{
    Json j = Json::object();
    j.set("state", Json::str(o.success ? "completed" : "failed"));
    if (!startedAt.empty()) j.set("started_at", Json::str(startedAt));
    if (!finishedAt.empty()) j.set("finished_at", Json::str(finishedAt));

    if (o.success)
    {
        // The metrics brae itself produced, passed through as an object. If they are unreadable we send none
        // rather than inventing a shape -- a wrong number is worse than a missing one.
        Json parsed;
        std::string err;
        if (!o.resultJson.empty() && Json::parse(o.resultJson, parsed, err)
            && parsed.type() == Json::Type::Object)
            j.set("result", std::move(parsed));
        else
            j.set("result", Json());
        j.set("error", Json());
    }
    else
    {
        j.set("result", Json());
        j.set("error", Json::str(o.error.empty() ? "unknown_failure" : o.error));
    }
    return j.dump();
}

// ---- response parsing -----------------------------------------------------------------------------------------

namespace {

// Pull the error code out of the standard envelope, so the agent can act on `code` and log something useful
// when it cannot. Never returns the raw body: it may be an HTML error page from a proxy.
std::string errorCodeOf(const HttpResponse& res)
{
    Json j;
    std::string err;
    if (Json::parse(res.body, j, err) && j["error"].type() == Json::Type::Object)
    {
        const std::string code = j["error"]["code"].asString();
        if (!code.empty()) return code;
    }
    return "http_" + std::to_string(res.status);
}

std::string transportOrStatus(const HttpResponse& res)
{
    return res.transportFailed() ? ("transport: " + res.error) : errorCodeOf(res);
}

}  // namespace

RegisterResult parseRegisterResponse(const HttpResponse& res)
{
    RegisterResult r;
    if (!res.ok()) { r.error = transportOrStatus(res); return r; }

    Json j;
    std::string err;
    if (!Json::parse(res.body, j, err)) { r.error = "unreadable register reply: " + err; return r; }

    r.nodeId = j["node_id"].asString();
    r.nodeToken = j["node_token"].asString();
    if (r.nodeId.empty() || r.nodeToken.empty())
    {
        // A 201 without an identity is a server we cannot work with. Failing here means `brae node register`
        // rolls back rather than installing a service that can never authenticate.
        r.error = "register reply has no node_id/node_token";
        return r;
    }
    r.snapshotIntervalIdleS = clampSnapshotInterval(j["snapshot_interval_idle_s"].asInt(10));
    r.snapshotIntervalRunningS = clampSnapshotInterval(j["snapshot_interval_running_s"].asInt(5));
    r.liveUrl = j["live_url"].asString();
    r.ok = true;
    return r;
}

SnapshotResult parseSnapshotResponse(const HttpResponse& res)
{
    SnapshotResult r;
    if (res.status == 401)
    {
        r.authRejected = true;
        r.error = "the control plane rejected this node's token";
        return r;
    }
    if (!res.ok()) { r.error = transportOrStatus(res); return r; }

    Json j;
    std::string err;
    if (!Json::parse(res.body, j, err)) { r.error = "unreadable snapshot reply: " + err; return r; }

    r.acceptsJobs = j["accepts_jobs"].asBool(true);
    r.snapshotIntervalS = clampSnapshotInterval(j["snapshot_interval_s"].asInt(10));

    const Json& a = j["assignment"];
    if (a.type() == Json::Type::Object)
    {
        // An assignment we cannot fully understand is left untaken. It expires server-side and is offered
        // elsewhere, which is strictly better than this node accepting work it cannot run.
        const JobType type = jobTypeFromString(a["type"].asString());
        const std::string sample = a["params"]["sample"].asString();
        const std::string jobId = a["job_id"].asString();
        if (type != JobType::Unknown && !jobId.empty() && validSampleName(sample))
        {
            r.assignment.present = true;
            r.assignment.jobId = jobId;
            r.assignment.type = type;
            r.assignment.sample = sample;
            r.assignment.deadline = a["deadline"].asString();
        }
        else
        {
            r.assignment.present = false;
        }
    }
    r.ok = true;
    return r;
}

SimpleResult parseSimpleResponse(const HttpResponse& res)
{
    SimpleResult r;
    if (res.status == 401) { r.authRejected = true; r.error = "token rejected"; return r; }
    if (res.status == 409) { r.conflict = true; r.error = errorCodeOf(res); return r; }
    if (!res.ok()) { r.error = transportOrStatus(res); return r; }
    r.ok = true;
    return r;
}

// ---- calls --------------------------------------------------------------------------------------------------

JobView parseJobView(const HttpResponse& res)
{
    JobView v;
    if (!res.ok())
    {
        // The message, not the code: this one is read by a person at a terminal, and "'gb' is too short"
        // helps where "invalid_request" does not. Falls back to the code, then to the status.
        Json j;
        std::string parseError;
        if (!res.transportFailed() && Json::parse(res.body, j, parseError)
            && j["error"].type() == Json::Type::Object)
        {
            const std::string message = j["error"]["message"].asString();
            v.error = message.empty() ? transportOrStatus(res) : message;
        }
        else
        {
            v.error = transportOrStatus(res);
        }
        return v;
    }

    Json j;
    std::string err;
    if (!Json::parse(res.body, j, err)) { v.error = "unreadable job reply: " + err; return v; }

    v.ok = true;
    v.jobId = j["job_id"].asString();
    v.state = j["state"].asString();
    v.sample = j["sample"].asString();
    v.nodeId = j["node_id"].asString();
    v.requestedGpuModel = j["requested_gpu_model"].asString();
    v.jobError = j["error"].asString();
    v.terminal = v.state == "completed" || v.state == "failed" || v.state == "abandoned";
    return v;
}


RegisterResult ApiClient::registerNode(const RegisterRequest& r, const std::string& enrollmentToken)
{
    HttpRequest req;
    req.method = "POST";
    req.url = urlJoin(base_, "/v1/nodes/register");
    req.body = buildRegisterBody(r);
    req.bearer = enrollmentToken;
    req.timeoutSeconds = 30;                      // registration also creates a NetBird peer server-side
    return parseRegisterResponse(http_.send(req));
}

SnapshotResult ApiClient::sendSnapshot(const std::string& nodeId, const std::string& token,
                                       const SnapshotRequest& s)
{
    HttpRequest req;
    req.method = "POST";
    req.url = urlJoin(base_, "/v1/nodes/" + nodeId + "/snapshot");
    req.body = buildSnapshotBody(s);
    req.bearer = token;
    req.timeoutSeconds = 15;                      // must be well under the offline threshold
    return parseSnapshotResponse(http_.send(req));
}

JobView ApiClient::queueJob(const std::string& sample, const std::string& gpu, const std::string& adminToken)
{
    Json body = Json::object();
    body.set("sample", Json::str(sample));
    if (!gpu.empty()) body.set("gpu", Json::str(gpu));

    HttpRequest req;
    req.method = "POST";
    req.url = urlJoin(base_, "/v1/jobs");
    req.body = body.dump();
    req.bearer = adminToken;
    return parseJobView(http_.send(req));
}

JobView ApiClient::getJob(const std::string& jobId, const std::string& adminToken)
{
    HttpRequest req;
    req.method = "GET";
    req.url = urlJoin(base_, "/v1/jobs/" + jobId);
    req.bearer = adminToken;
    return parseJobView(http_.send(req));
}

SimpleResult ApiClient::acceptJob(const std::string& nodeId, const std::string& token, const std::string& jobId)
{
    HttpRequest req;
    req.method = "POST";
    req.url = urlJoin(base_, "/v1/jobs/" + jobId + "/accept?node_id=" + nodeId);
    req.bearer = token;
    return parseSimpleResponse(http_.send(req));
}

SimpleResult ApiClient::reportResult(const std::string& nodeId, const std::string& token, const JobOutcome& o,
                                     const std::string& startedAt, const std::string& finishedAt)
{
    HttpRequest req;
    req.method = "POST";
    req.url = urlJoin(base_, "/v1/jobs/" + o.jobId + "/result?node_id=" + nodeId);
    req.body = buildResultBody(o, startedAt, finishedAt);
    req.bearer = token;
    return parseSimpleResponse(http_.send(req));
}

SimpleResult ApiClient::unregisterNode(const std::string& nodeId, const std::string& token)
{
    HttpRequest req;
    req.method = "POST";
    req.url = urlJoin(base_, "/v1/nodes/" + nodeId + "/unregister");
    req.bearer = token;
    return parseSimpleResponse(http_.send(req));
}

}  // namespace brae::node
