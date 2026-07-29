#pragma once
// Minimal JSON for brae-agent: enough to build a snapshot and read an orchestrator reply, and nothing more.
//
// Hand-written rather than vendored because the agent's whole dependency budget is libcurl and dlopen'd NVML,
// and because the parser reads UNTRUSTED input -- every response from the control plane, which we must assume
// can be hostile (docs/10-security.md, rule 7 in reverse). So it is bounded on purpose:
//
//   * nesting depth        -- a deeply nested array would otherwise recurse the stack into a crash
//   * total input length   -- a reply that never ends must not become unbounded memory
//   * no duplicate-key or comment extensions, no trailing commas: strict JSON or an error
//
// It parses what it needs and refuses the rest. There is no "best effort" mode: a reply we cannot read is a
// reply we do not act on.
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace brae::node {

class Json
{
public:
    enum class Type { Null, Bool, Number, String, Array, Object };

    Json() = default;
    static Json object() { Json j; j.type_ = Type::Object; return j; }
    static Json array()  { Json j; j.type_ = Type::Array;  return j; }
    static Json str(std::string v)  { Json j; j.type_ = Type::String; j.str_ = std::move(v); return j; }
    static Json num(double v)       { Json j; j.type_ = Type::Number; j.num_ = v; return j; }
    static Json boolean(bool v)     { Json j; j.type_ = Type::Bool;   j.bool_ = v; return j; }

    Type type() const { return type_; }
    bool isNull() const { return type_ == Type::Null; }

    // Accessors. Every one takes a default: a missing or wrong-typed field is a normal case when talking to a
    // service that may be a version ahead of us, not an exception.
    std::string asString(const std::string& dflt = "") const { return type_ == Type::String ? str_ : dflt; }
    double asNumber(double dflt = 0) const { return type_ == Type::Number ? num_ : dflt; }
    long long asInt(long long dflt = 0) const
    { return type_ == Type::Number ? static_cast<long long>(num_) : dflt; }
    bool asBool(bool dflt = false) const { return type_ == Type::Bool ? bool_ : dflt; }

    // Object/array access. Absent -> a Null Json, so `reply["job"]["progress"].asInt(0)` is safe on any shape.
    const Json& operator[](const std::string& key) const;
    const Json& operator[](std::size_t i) const;
    bool has(const std::string& key) const;
    std::size_t size() const;

    void set(const std::string& key, Json value);
    void push(Json value);

    std::string dump() const;

    // Parse. Returns false and fills `error` rather than throwing: the caller is a retry loop, and an
    // unparseable reply is a thing to log and retry, not to crash the daemon over.
    static bool parse(const std::string& text, Json& out, std::string& error);

    static constexpr std::size_t kMaxDepth = 32;
    static constexpr std::size_t kMaxInput = 1u << 20;   // 1 MiB: far more than any reply we define

private:
    Type type_ = Type::Null;
    bool bool_ = false;
    double num_ = 0;
    std::string str_;
    std::vector<Json> arr_;
    std::vector<std::pair<std::string, Json>> obj_;      // insertion-ordered: dumps are reproducible
};

std::string jsonEscape(const std::string& s);

}  // namespace brae::node
