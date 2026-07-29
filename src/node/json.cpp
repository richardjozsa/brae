#include "json.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace brae::node {
namespace {

const Json kNull;

struct Parser
{
    const std::string& text;
    std::size_t pos = 0;
    std::string error;

    explicit Parser(const std::string& t) : text(t) {}

    void skipSpace()
    {
        while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\t' || text[pos] == '\n' || text[pos] == '\r'))
            ++pos;
    }
    bool fail(const std::string& why)
    {
        if (error.empty()) error = why + " at byte " + std::to_string(pos);
        return false;
    }
    bool eat(char c)
    {
        skipSpace();
        if (pos < text.size() && text[pos] == c) { ++pos; return true; }
        return fail(std::string("expected '") + c + "'");
    }

    bool parseValue(Json& out, std::size_t depth);

    bool parseString(std::string& out)
    {
        if (!eat('"')) return false;
        out.clear();
        while (pos < text.size())
        {
            const char c = text[pos++];
            if (c == '"') return true;
            if (c != '\\') { out += c; continue; }
            if (pos >= text.size()) return fail("string ends inside an escape");
            const char e = text[pos++];
            switch (e)
            {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u':
                {
                    // \uXXXX. Only the BMP, encoded as UTF-8; surrogate pairs are joined if both halves are
                    // present. Nothing we read carries non-ASCII today, but a GPU model name could.
                    if (pos + 4 > text.size()) return fail("truncated \\u escape");
                    unsigned cp = 0;
                    for (int i = 0; i < 4; ++i)
                    {
                        const char h = text[pos + i];
                        cp <<= 4;
                        if (h >= '0' && h <= '9') cp |= unsigned(h - '0');
                        else if (h >= 'a' && h <= 'f') cp |= unsigned(h - 'a' + 10);
                        else if (h >= 'A' && h <= 'F') cp |= unsigned(h - 'A' + 10);
                        else return fail("bad hex in \\u escape");
                    }
                    pos += 4;
                    if (cp >= 0xD800 && cp <= 0xDBFF && pos + 6 <= text.size()
                        && text[pos] == '\\' && text[pos + 1] == 'u')
                    {
                        unsigned lo = 0;
                        bool ok = true;
                        for (int i = 0; i < 4 && ok; ++i)
                        {
                            const char h = text[pos + 2 + i];
                            lo <<= 4;
                            if (h >= '0' && h <= '9') lo |= unsigned(h - '0');
                            else if (h >= 'a' && h <= 'f') lo |= unsigned(h - 'a' + 10);
                            else if (h >= 'A' && h <= 'F') lo |= unsigned(h - 'A' + 10);
                            else ok = false;
                        }
                        if (ok && lo >= 0xDC00 && lo <= 0xDFFF)
                        {
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            pos += 6;
                        }
                    }
                    if (cp < 0x80) out += char(cp);
                    else if (cp < 0x800) { out += char(0xC0 | (cp >> 6)); out += char(0x80 | (cp & 0x3F)); }
                    else if (cp < 0x10000)
                    {
                        out += char(0xE0 | (cp >> 12));
                        out += char(0x80 | ((cp >> 6) & 0x3F));
                        out += char(0x80 | (cp & 0x3F));
                    }
                    else
                    {
                        out += char(0xF0 | (cp >> 18));
                        out += char(0x80 | ((cp >> 12) & 0x3F));
                        out += char(0x80 | ((cp >> 6) & 0x3F));
                        out += char(0x80 | (cp & 0x3F));
                    }
                    break;
                }
                default: return fail("unknown escape");
            }
        }
        return fail("unterminated string");
    }
};

bool Parser::parseValue(Json& out, std::size_t depth)
{
    if (depth > Json::kMaxDepth) return fail("nested deeper than " + std::to_string(Json::kMaxDepth));
    skipSpace();
    if (pos >= text.size()) return fail("unexpected end of input");

    const char c = text[pos];
    if (c == '{')
    {
        ++pos;
        out = Json::object();
        skipSpace();
        if (pos < text.size() && text[pos] == '}') { ++pos; return true; }
        for (;;)
        {
            std::string key;
            if (!parseString(key)) return false;
            if (!eat(':')) return false;
            Json value;
            if (!parseValue(value, depth + 1)) return false;
            out.set(key, std::move(value));
            skipSpace();
            if (pos < text.size() && text[pos] == ',') { ++pos; continue; }
            return eat('}');
        }
    }
    if (c == '[')
    {
        ++pos;
        out = Json::array();
        skipSpace();
        if (pos < text.size() && text[pos] == ']') { ++pos; return true; }
        for (;;)
        {
            Json value;
            if (!parseValue(value, depth + 1)) return false;
            out.push(std::move(value));
            skipSpace();
            if (pos < text.size() && text[pos] == ',') { ++pos; continue; }
            return eat(']');
        }
    }
    if (c == '"')
    {
        std::string s;
        if (!parseString(s)) return false;
        out = Json::str(std::move(s));
        return true;
    }
    if (text.compare(pos, 4, "true") == 0)  { pos += 4; out = Json::boolean(true);  return true; }
    if (text.compare(pos, 5, "false") == 0) { pos += 5; out = Json::boolean(false); return true; }
    if (text.compare(pos, 4, "null") == 0)  { pos += 4; out = Json();               return true; }

    // number
    const char* start = text.c_str() + pos;
    char* end = nullptr;
    const double v = std::strtod(start, &end);
    if (end == start) return fail("not a value");
    if (!std::isfinite(v)) return fail("number is not finite");
    pos += static_cast<std::size_t>(end - start);
    out = Json::num(v);
    return true;
}

}  // namespace

const Json& Json::operator[](const std::string& key) const
{
    for (const auto& kv : obj_)
        if (kv.first == key) return kv.second;
    return kNull;
}

const Json& Json::operator[](std::size_t i) const
{
    return i < arr_.size() ? arr_[i] : kNull;
}

bool Json::has(const std::string& key) const
{
    for (const auto& kv : obj_)
        if (kv.first == key) return true;
    return false;
}

std::size_t Json::size() const
{
    if (type_ == Type::Array) return arr_.size();
    if (type_ == Type::Object) return obj_.size();
    return 0;
}

void Json::set(const std::string& key, Json value)
{
    type_ = Type::Object;
    for (auto& kv : obj_)
        if (kv.first == key) { kv.second = std::move(value); return; }
    obj_.emplace_back(key, std::move(value));
}

void Json::push(Json value)
{
    type_ = Type::Array;
    arr_.push_back(std::move(value));
}

std::string jsonEscape(const std::string& s)
{
    std::string o;
    o.reserve(s.size() + 8);
    for (const unsigned char c : s)
    {
        switch (c)
        {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n";  break;
            case '\r': o += "\\r";  break;
            case '\t': o += "\\t";  break;
            case '\b': o += "\\b";  break;
            case '\f': o += "\\f";  break;
            default:
                if (c < 0x20)
                {
                    char buf[8];
                    std::snprintf(buf, sizeof buf, "\\u%04x", c);
                    o += buf;
                }
                else o += char(c);
        }
    }
    return o;
}

std::string Json::dump() const
{
    switch (type_)
    {
        case Type::Null:   return "null";
        case Type::Bool:   return bool_ ? "true" : "false";
        case Type::String: return "\"" + jsonEscape(str_) + "\"";
        case Type::Number:
        {
            // Integral values print without a decimal point: the API's ints (ports, percentages, megabytes)
            // must not arrive as 91.0 and fail an int-typed schema on the far side.
            if (num_ == static_cast<long long>(num_))
                return std::to_string(static_cast<long long>(num_));
            char buf[40];
            std::snprintf(buf, sizeof buf, "%.10g", num_);
            return buf;
        }
        case Type::Array:
        {
            std::string o = "[";
            for (std::size_t i = 0; i < arr_.size(); ++i) { if (i) o += ','; o += arr_[i].dump(); }
            return o + "]";
        }
        case Type::Object:
        {
            std::string o = "{";
            bool first = true;
            for (const auto& kv : obj_)
            {
                if (!first) o += ',';
                first = false;
                o += "\"" + jsonEscape(kv.first) + "\":" + kv.second.dump();
            }
            return o + "}";
        }
    }
    return "null";
}

bool Json::parse(const std::string& text, Json& out, std::string& error)
{
    error.clear();
    if (text.size() > kMaxInput)
    {
        error = "input larger than " + std::to_string(kMaxInput) + " bytes";
        return false;
    }
    Parser p(text);
    Json value;
    if (!p.parseValue(value, 0)) { error = p.error; return false; }
    p.skipSpace();
    if (p.pos != text.size()) { error = "trailing data at byte " + std::to_string(p.pos); return false; }
    out = std::move(value);
    return true;
}

}  // namespace brae::node
