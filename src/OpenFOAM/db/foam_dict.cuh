#pragma once
// Minimal OpenFOAM dictionary reader. Parses a foam dict file (FoamFile header already stripped by
// TokenStream) into a nested key/value tree, with OpenFOAM's regex-keyword lookup semantics:
// a literal match wins; otherwise the LAST regex key that matches is used (mirrors
// Foam::dictionary::lookupEntry handling of wildcard keywords like "(k|epsilon)" and ".*").
// Sufficient for reading controlDict / fvSolution / transportProperties / turbulenceProperties.
#include "foam_token_reader.cuh"
#include <cctype>
#include <fstream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace brae {

// Compile an OpenFOAM keyType pattern to std::regex. OF patterns are POSIX-extended with an optional inline
// case-insensitive prefix "(?i)" (e.g. "(?i).*walls"); std::regex (ECMAScript) does not honour inline flags, so
// we strip the prefix and set std::regex::icase explicitly. Used everywhere a dict/boundary key may be a wildcard.
inline std::regex compileFoamRegex(const std::string& pat)
{
    auto flags = std::regex::ECMAScript;
    std::string p = pat;
    if (p.size() >= 4 && p.compare(0, 4, "(?i)") == 0)
    {
        flags |= std::regex::icase;
        p.erase(0, 4);
    }
    return std::regex(p, flags);
}

// Constraint patch types: their boundary field is determined by the mesh patch type itself (OF auto-selects the
// matching constraint fvPatchField), so a field file may omit them, exactly what #includeEtc setConstraintTypes
// provides as defaults. cf synthesises the entry from the patch type when no boundaryField entry is present.
inline bool isConstraintPatchType(const std::string& t)
{
    return t == "empty" || t == "symmetryPlane" || t == "symmetry" || t == "wedge" || t == "cyclic"
        || t == "cyclicAMI" || t == "cyclicACMI" || t == "cyclicSlip" || t == "processor"
        || t == "processorCyclic" || t == "nonuniformTransformCyclic";
    // NOTE: "overset" is deliberately NOT here. It is a coupled patch whose fvPatchField cannot be
    // synthesised from the mesh type -- treating it as a constraint made an unsupported overset case run
    // silently. buildPatches refuses it outright (fv_patch.cu).
}

struct FoamDict
{
    std::vector<std::pair<std::string, std::vector<std::string>>> leaves;   // key -> value tokens
    std::vector<std::pair<std::string, FoamDict>>                 subs;     // key -> subdict

    // Every key this dict was ever ASKED for. The point is the complement: a key that is in the file but
    // never in here is an input brae read off disk and then ignored -- the single failure mode that has
    // produced almost every defect in this project (present, parsed, never applied). Recorded on the
    // lookup path itself so it cannot drift from what the code actually consumes; `mutable` because
    // asking a question about a dict is logically const.
    mutable std::set<std::string> queried;

    const FoamDict* subDict(const std::string& key) const
    {
        queried.insert(key);
        for (const auto& s : subs)
            if (s.first == key) return &s.second;          // literal wins (duplicates were merged at parse)
        const FoamDict* hit = nullptr;
        for (const auto& s : subs)                                              // else last regex match
        {
            if (s.first == key) continue;
            try
            {
                if (std::regex_match(key, compileFoamRegex(s.first))) { hit = &s.second; queried.insert(s.first); }
            }
            catch (...) {}
        }
        return hit;
    }
    // Regex-aware leaf lookup: literal first, else last matching wildcard key (OF semantics).
    const std::vector<std::string>* find(const std::string& name) const
    {
        queried.insert(name);
        const std::vector<std::string>* lit = nullptr;      // LAST literal match wins, as in OpenFOAM
        for (const auto& l : leaves)
            if (l.first == name) lit = &l.second;
        if (lit) return lit;
        const std::vector<std::string>* hit = nullptr;
        for (const auto& l : leaves)                                                // else last regex match
        {
            if (l.first == name) continue;
            try
            {
                // A regex key that MATCHED counts as consumed -- `"(k|omega|e)" 1e-4` is one entry
                // answering three questions, and reporting it unread would be noise, not a gap.
                if (std::regex_match(name, compileFoamRegex(l.first))) { hit = &l.second; queried.insert(l.first); }
            }
            catch (...) {}
        }
        return hit;
    }
    bool found(const std::string& name) const { return find(name) != nullptr; }

    // Value extraction: scalar/int is the LAST value token (handles `key [dims] value`); word is first.
    scalar scalarOr(const std::string& name, scalar def) const
    {
        const auto* v = find(name);
        return (v && !v->empty()) ? std::stod(v->back()) : def;
    }
    int intOr(const std::string& name, int def) const
    {
        const auto* v = find(name);
        return (v && !v->empty()) ? std::stoi(v->back()) : def;
    }
    std::string wordOr(const std::string& name, const std::string& def) const
    {
        const auto* v = find(name);
        return (v && !v->empty()) ? v->front() : def;
    }
    // List entries (e.g. liftDir (0 1 0); patches (upperWall lowerWall);). Parens are stripped; scalars are the
    // numeric tokens, words are the non-empty bare tokens. Returns def when the key is absent/empty.
    std::vector<scalar> scalarListOr(const std::string& name, const std::vector<scalar>& def) const
    {
        const auto* t = find(name);
        if (!t) return def;
        std::string joined;
        for (const auto& s : *t) joined += " " + s;
        static const std::regex re(R"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)");
        std::vector<scalar> v;
        for (std::sregex_iterator it(joined.begin(), joined.end(), re), e; it != e; ++it) v.push_back(std::stod(it->str()));
        return v.empty() ? def : v;
    }
    std::vector<std::string> wordListOr(const std::string& name, const std::vector<std::string>& def) const
    {
        const auto* t = find(name);
        if (!t) return def;
        std::vector<std::string> out;
        for (const auto& s : *t)
        {
            std::string w;
            for (char ch : s) if (ch != '(' && ch != ')') w += ch;
            if (!w.empty()) out.push_back(w);
        }
        return out.empty() ? def : out;
    }
};

// Parse a dict body from the token cursor. `top` = file top level (read to EOF); else read to '}'.
inline FoamDict parseDictBody(TokenStream& ts, bool top)
{
    FoamDict d;
    while (!ts.eof())
    {
        if (ts.peek() == "}")
        {
            if (!top) ts.next();
            break;
        }
        const std::string key = ts.next();
        if (ts.eof()) break;
        if (ts.peek() == "{")
        {
            ts.next();
            {
                // A REPEATED sub-dictionary MERGES into the existing one, later keys winning -- OpenFOAM's
                // documented behaviour and not the same rule as for leaf entries (those simply override).
                // gasMixing's thermophysicalProperties relies on it: a full `thermoType { type hePsiThermo;
                // ... equationOfState perfectGas; ... }` followed by `thermoType { type heRhoThermo; }`,
                // which foamDictionary resolves to heRhoThermo + perfectGas. Replacing wholesale would have
                // left a thermoType with nothing but `type`, and brae would refuse a case OpenFOAM runs.
                FoamDict body = parseDictBody(ts, false);
                FoamDict* existing = nullptr;
                for (auto& sd : d.subs)
                    if (sd.first == key) existing = &sd.second;
                if (!existing)
                {
                    d.subs.emplace_back(key, std::move(body));
                }
                else
                {
                    for (auto& l : body.leaves)
                    {
                        bool replaced = false;
                        for (auto& e : existing->leaves)
                            if (e.first == l.first) { e.second = l.second; replaced = true; break; }
                        if (!replaced) existing->leaves.push_back(l);
                    }
                    for (auto& sb : body.subs)
                    {
                        bool replaced = false;
                        for (auto& e : existing->subs)
                            if (e.first == sb.first) { e.second = sb.second; replaced = true; break; }
                        if (!replaced) existing->subs.push_back(sb);
                    }
                }
            }
        }
        else
        {
            std::vector<std::string> vals;
            while (!ts.eof() && ts.peek() != ";" && ts.peek() != "}" && ts.peek() != "{") vals.push_back(ts.next());
            if (!ts.eof() && ts.peek() == ";") ts.next();
            d.leaves.emplace_back(key, std::move(vals));
        }
    }
    return d;
}

inline FoamDict readDict(const std::string& path)
{
    // expandVars=true: expand $macros in constant/system dicts too, e.g. $RASturbModel / $nu pulled in from an
    // #included system/include/caseDefinition. OF expands $variables in every dictionary (a field file is just a
    // dictionary), so match that here rather than only in the field reader and fvSchemes.
    TokenStream ts(path, /*expandVars=*/true);
    return parseDictBody(ts, true);
}

// OpenFOAM dictionary $variable expansion (db/dictionary expandVariable): e.g. fvSchemes
//   turbulence       bounded Gauss limitedLinear 1;
//   div(phi,k)       $turbulence;
// Strips //... and /*...*/ comments, builds a map of simple "name value... ;" entries (name = plain identifier,
// value not a sub-dict), then substitutes $name / ${name} in the text with that value (a few passes for nesting).
// Faithful to expandVariable for the sibling/ancestor (non-scoped) case used in fvSchemes; preserves newlines so a
// downstream per-line scan is unaffected. Scoped $:abs / $..parent forms are not handled (unused in fvSchemes).
inline std::string expandDictVariables(const std::string& rawIn)
{
    // 1. strip comments
    std::string raw;
    raw.reserve(rawIn.size());
    for (std::size_t i = 0; i < rawIn.size(); )
    {
        if (rawIn[i] == '/' && i + 1 < rawIn.size() && rawIn[i + 1] == '/')
        {
            while (i < rawIn.size() && rawIn[i] != '\n') ++i;
        }
        else if (rawIn[i] == '/' && i + 1 < rawIn.size() && rawIn[i + 1] == '*')
        {
            i += 2;
            while (i + 1 < rawIn.size() && !(rawIn[i] == '*' && rawIn[i + 1] == '/')) ++i;
            i += 2;
        }
        else raw += rawIn[i++];
    }
    // 2. tokenize with { } ; as separate tokens (parens stay inside their token, e.g. div(phi,U))
    std::string padded;
    padded.reserve(raw.size() * 2);
    for (char c : raw)
    {
        if (c == '{' || c == '}' || c == ';') { padded += ' '; padded += c; padded += ' '; }
        else padded += c;
    }
    std::vector<std::string> toks;
    {
        std::istringstream is(padded);
        std::string t;
        while (is >> t) toks.push_back(t);
    }
    // 3. build the variable map: a "name value... ;" entry whose key is a plain identifier and value isn't a subdict
    auto isIdent = [](const std::string& s)
    {
        if (s.empty() || !(std::isalpha((unsigned char)s[0]) || s[0] == '_')) return false;
        for (char c : s) if (!(std::isalnum((unsigned char)c) || c == '_')) return false;
        return true;
    };
    std::map<std::string, std::string> vars;
    bool atKey = true;
    for (std::size_t i = 0; i < toks.size(); )
    {
        const std::string& tk = toks[i];
        if (tk == "{" || tk == "}" || tk == ";")
        {
            atKey = true;
            ++i;
            continue;
        }
        if (!atKey)
        {
            ++i;
            continue;
        }
        if (i + 1 < toks.size() && toks[i + 1] == "{")   // SUBDICT (e.g. intakeType1 { ... } or divSchemes { ... }):
        {
            std::size_t j = i + 2;
            int depth = 1;
            std::string val;   // capture its CONTENT so $intakeType1 -> entries (OF dict-merge)
            while (j < toks.size() && depth > 0)
            {
                if (toks[j] == "{") ++depth;
                else if (toks[j] == "}") { --depth; if (depth == 0) { ++j; break; } }
                if (depth > 0) { if (!val.empty()) val += ' '; val += toks[j]; }
                ++j;
            }
            if (isIdent(tk)) vars[tk] = val;
            // DESCEND into the block (step past name + "{") rather than skip it, so sibling entries DEFINED inside
            // (e.g. `turbulence ...;` inside divSchemes, referenced as $turbulence by div(phi,k)) also register as vars.
            i += 2;
            atKey = true;
            continue;
        }
        std::string val;
        std::size_t j = i + 1;
        for (; j < toks.size() && toks[j] != ";" && toks[j] != "{" && toks[j] != "}"; ++j) { if (!val.empty()) val += ' '; val += toks[j]; }
        // Skip a self-reference like `z0 $z0;` (an inner-scope entry that pulls from an outer variable of the SAME
        // name -- OF scoping). In brae's flat var map this would overwrite the real outer value with an unresolvable
        // self-ref, so keep the outer definition (e.g. `z0 uniform 0.1;` from an #include'd ABLConditions).
        if (isIdent(tk) && val != "$" + tk && val != "${" + tk + "}") vars[tk] = val;
        i = j;
        atKey = true;
    }
    // 4. substitute $name / ${name} in the (comment-stripped) text, preserving newlines; a few passes for nesting
    std::string text = raw;
    for (int pass = 0; pass < 5; ++pass)
    {
        std::string out;
        out.reserve(text.size());
        bool changed = false;
        for (std::size_t i = 0; i < text.size(); )
        {
            if (text[i] == '$')
            {
                std::size_t j = i + 1;
                bool brace = false;
                if (j < text.size() && text[j] == '{') { brace = true; ++j; }
                const std::size_t s = j;
                while (j < text.size() && (std::isalnum((unsigned char)text[j]) || text[j] == '_')) ++j;
                const std::string name = text.substr(s, j - s);
                if (brace && j < text.size() && text[j] == '}') ++j;
                const auto it = vars.find(name);
                if (!name.empty() && it != vars.end())
                {
                    out += it->second;
                    i = j;
                    changed = true;
                    continue;
                }
            }
            out += text[i++];
        }
        text = std::move(out);
        if (!changed) break;
    }
    return text;
}
inline std::string readFileExpanded(const std::string& path)
{
    std::ifstream f(path);
    std::stringstream ss;
    ss << f.rdbuf();
    return expandDictVariables(ss.str());
}

} // namespace brae
