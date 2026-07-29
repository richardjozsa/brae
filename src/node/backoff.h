#pragma once
// Retry timing, as pure arithmetic so it can be tested without waiting.
//
// Two different schedules, because two different failures:
//
//   snapshots     do NOT back off and do NOT queue. The next snapshot supersedes the one that failed, so
//                 retrying at the normal interval is both correct and self-limiting. Queueing missed snapshots
//                 would mean a node returning from an outage floods the API with stale state nobody wants.
//
//   job results   DO back off, and never give up. A result is the only record that the work happened; losing
//                 it strands the job until its deadline and wastes the whole run. Exponential to a ceiling,
//                 so a long outage settles into a slow retry rather than a hot loop.
#include <cstdint>

namespace brae::node {

// 2, 4, 8, ... capped. attempt is 0-based: attempt 0 is the first retry after the first failure.
constexpr int kResultBackoffBaseSeconds = 2;
constexpr int kResultBackoffCapSeconds = 60;

inline int resultBackoffSeconds(int attempt)
{
    if (attempt < 0) attempt = 0;
    long long delay = kResultBackoffBaseSeconds;
    for (int i = 0; i < attempt && delay < kResultBackoffCapSeconds; ++i) delay *= 2;
    return static_cast<int>(delay > kResultBackoffCapSeconds ? kResultBackoffCapSeconds : delay);
}

// The server tells the agent how often to report; the agent still clamps it. A control plane that says "every
// 0 seconds" -- by bug or by malice -- must not turn every contributor's machine into a request flood, and one
// that says "every hour" must not silently take a node off the live page.
constexpr int kMinSnapshotIntervalSeconds = 5;
constexpr int kMaxSnapshotIntervalSeconds = 60;

inline int clampSnapshotInterval(long long requested)
{
    if (requested < kMinSnapshotIntervalSeconds) return kMinSnapshotIntervalSeconds;
    if (requested > kMaxSnapshotIntervalSeconds) return kMaxSnapshotIntervalSeconds;
    return static_cast<int>(requested);
}

}  // namespace brae::node
