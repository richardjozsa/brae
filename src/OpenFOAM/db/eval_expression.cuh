#pragma once
// #eval{...} -- OF src/OpenFOAM/expressions (exprDriver / exprScanner).
//
// OF lets a dictionary compute a value inline:
//     Uinlet  #eval{ cos(degToRad($AoA)) };
//     k       #eval{ 1.5*sqr(0.00086) };
//     nut     #eval{ 3.0/1520000.0 };
// (pimpleFoam/LES/NACA4412's 0/ fields). brae refused these outright -- correctly, since guessing a
// value would be worse -- but that blocked the case at the first field read.
//
// SCOPE. This evaluates SCALAR expressions: numbers, + - * / and unary minus, parentheses, and the
// function set OF's scanner exposes. It is NOT OF's full expression engine: OF also evaluates FIELD
// expressions (per-cell, referencing pos(), vol(), other fields), and those are refused by name rather
// than mis-evaluated -- the whole reason the previous blanket refusal existed.
//
// $macros are expanded BEFORE this runs, by the reader's existing readFileExpanded, so `$AoA` has
// already become a literal by the time an expression reaches here.
//
// A malformed or unsupported expression THROWS. It never falls back to a default: an #eval that
// silently became 0 would be exactly the plausible-wrong-answer failure this codebase refuses.

#include "cf_types.cuh"
#include <cctype>
#include <cmath>
#include <stdexcept>
#include <string>

namespace brae {

namespace detail {

// Recursive-descent over the expression text. `i` is the cursor, advanced in place.
struct EvalParser
{
    const std::string& s;
    std::size_t        i = 0;
    const std::string& what;   // the original text, for error messages

    void skip() { while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i; }

    [[noreturn]] void fail(const std::string& msg) const
    {
        throw std::runtime_error("brae: cannot evaluate #eval{" + what + "}: " + msg);
    }

    scalar parseExpr()                       // + and -
    {
        scalar v = parseTerm();
        for (;;)
        {
            skip();
            if (i < s.size() && s[i] == '+') { ++i; v += parseTerm(); }
            else if (i < s.size() && s[i] == '-') { ++i; v -= parseTerm(); }
            else return v;
        }
    }

    scalar parseTerm()                       // * and /
    {
        scalar v = parseUnary();
        for (;;)
        {
            skip();
            if (i < s.size() && s[i] == '*') { ++i; v *= parseUnary(); }
            else if (i < s.size() && s[i] == '/')
            {
                ++i;
                const scalar d = parseUnary();
                if (d == 0) fail("division by zero");
                v /= d;
            }
            else return v;
        }
    }

    scalar parseUnary()
    {
        skip();
        if (i < s.size() && s[i] == '-') { ++i; return -parseUnary(); }
        if (i < s.size() && s[i] == '+') { ++i; return  parseUnary(); }
        return parseAtom();
    }

    scalar parseAtom()
    {
        skip();
        if (i >= s.size()) fail("unexpected end of expression");
        if (s[i] == '(')
        {
            ++i;
            const scalar v = parseExpr();
            skip();
            if (i >= s.size() || s[i] != ')') fail("missing ')'");
            ++i;
            return v;
        }
        if (std::isdigit(static_cast<unsigned char>(s[i])) || s[i] == '.')
        {
            char* end = nullptr;
            const scalar v = static_cast<scalar>(std::strtod(s.c_str() + i, &end));
            if (end == s.c_str() + i) fail("malformed number");
            i = static_cast<std::size_t>(end - s.c_str());
            return v;
        }
        if (std::isalpha(static_cast<unsigned char>(s[i])) || s[i] == '_')
        {
            const std::size_t b = i;
            while (i < s.size() && (std::isalnum(static_cast<unsigned char>(s[i])) || s[i] == '_')) ++i;
            const std::string name = s.substr(b, i - b);
            skip();
            // `pi` appears BOTH bare and as `pi()` (LES/planeChannel uses the latter). Accept either.
            if (name == "pi")
            {
                if (i < s.size() && s[i] == '(')
                {
                    ++i; skip();
                    if (i >= s.size() || s[i] != ')') fail("pi() takes no arguments");
                    ++i;
                }
                return static_cast<scalar>(M_PI);
            }
            if (i >= s.size() || s[i] != '(')
                fail("unknown symbol '" + name + "' (a $macro that did not expand, or a field "
                     "expression -- brae evaluates scalar expressions only)");
            ++i;
            const scalar a = parseExpr();
            skip();
            scalar b2 = 0;
            bool haveSecond = false;
            if (i < s.size() && s[i] == ',') { ++i; b2 = parseExpr(); haveSecond = true; skip(); }
            if (i >= s.size() || s[i] != ')') fail("missing ')' after " + name);
            ++i;
            // OF's scanner vocabulary (exprScanner): the standard maths set plus OF's own sqr/magSqr/
            // degToRad/radToDeg. Anything outside it is named, not approximated.
            if (name == "sqr")       return a*a;
            if (name == "magSqr")    return a*a;
            if (name == "sqrt")      return std::sqrt(a);
            if (name == "cbrt")      return std::cbrt(a);
            if (name == "mag")       return std::fabs(a);
            if (name == "sin")       return std::sin(a);
            if (name == "cos")       return std::cos(a);
            if (name == "tan")       return std::tan(a);
            if (name == "asin")      return std::asin(a);
            if (name == "acos")      return std::acos(a);
            if (name == "atan")      return std::atan(a);
            if (name == "exp")       return std::exp(a);
            if (name == "log")       return std::log(a);
            if (name == "log10")     return std::log10(a);
            if (name == "floor")     return std::floor(a);
            if (name == "ceil")      return std::ceil(a);
            if (name == "round")     return std::round(a);
            if (name == "degToRad")  return a*static_cast<scalar>(M_PI)/scalar(180);
            if (name == "radToDeg")  return a*scalar(180)/static_cast<scalar>(M_PI);
            if (haveSecond)
            {
                if (name == "pow")   return std::pow(a, b2);
                if (name == "atan2") return std::atan2(a, b2);
                if (name == "hypot") return std::hypot(a, b2);
                if (name == "min")   return std::fmin(a, b2);
                if (name == "max")   return std::fmax(a, b2);
            }
            fail("function '" + name + "' is not implemented");
        }
        fail(std::string("unexpected character '") + s[i] + "'");
    }
};

}   // namespace detail

// Evaluate the body of an #eval{...}. Throws on anything it cannot evaluate exactly.
inline scalar evalExpression(const std::string& body)
{
    detail::EvalParser p{body, 0, body};
    const scalar v = p.parseExpr();
    p.skip();
    if (p.i != body.size()) p.fail("trailing characters after the expression");
    return v;
}

}   // namespace brae
