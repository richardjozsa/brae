#pragma once
// SHA-256, so the agent can hash a GPU UUID without pulling in OpenSSL.
//
// The one thing it is used for matters: a GPU UUID is a hardware serial, and the control plane stores only its
// hash (docs/01-contracts.md). Hashing at the source means the raw UUID never reaches a struct that gets
// serialized, so it cannot leak by someone later adding a field to a snapshot.
//
// Reference implementation of FIPS 180-4, checked against the standard vectors in tests/test_node_agent.cpp.
#include <cstdint>
#include <string>

namespace brae::node {

std::string sha256Hex(const std::string& data);

}  // namespace brae::node
