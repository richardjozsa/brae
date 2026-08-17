// #eval and its $variable references -- OF's expressions/exprDriver plus dictionary variable scoping.
//
// A dictionary can compute an entry from another entry:
//     endTime    2;
//     timeStart  #eval #{ 1.0/3.0 * ${/endTime} #};      (LES/planeChannel's controlDict)
// Three separate things have to work for that line: the `#{ ... #}` verbatim delimiters, the arithmetic,
// and `${/endTime}` -- the SCOPED reference, meaning "endTime at the file's ROOT scope" rather than the
// enclosing sub-dictionary. brae expanded `$name` and `${name}` but not the scoped spellings, so the
// leading '/' was left in the text and reached the expression parser as an operator: the run stopped at
// "cannot evaluate #eval{ 1.0/3.0 * ${/endTime} }: unexpected character".
//
// WHY IT IS EXPANSION AND NOT EVALUATION. brae's variable map is FLAT -- every entry, at whatever depth,
// registered under its own name -- so a root-scoped lookup is the ordinary lookup with the '/' dropped.
// That is only true for a SINGLE-level reference; `${../x}` and `${a/b/c}` need a real scope tree and are
// deliberately left unexpanded, which leg 4 pins: an unresolvable reference must stay unresolved and be
// refused downstream, never resolved to whatever entry happens to share the last name.
#include "foam_dict.cuh"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>

using namespace brae;

namespace {
int failures = 0;

void check(bool ok, const char* what)
{
    if (!ok) { std::printf("  FAIL: %s\n", what); ++failures; }
    else       std::printf("  ok:   %s\n", what);
}

FoamDict parse(const std::string& body)
{
    const std::string path = "test_eval_scoped.tmpdict";
    { std::ofstream f(path); f << body; }
    FoamDict d = readDict(path);
    std::remove(path.c_str());
    return d;
}
} // namespace

int main()
{
    std::printf("== #eval with scoped $variables ==\n");

    // ---- Leg 1: the planeChannel line, verbatim ----------------------------------------------------
    {
        const FoamDict d = parse("endTime 6;\n"
                                 "functions { sample { timeStart #eval #{ 1.0/3.0 * ${/endTime} #}; } }\n");
        const FoamDict* f = d.subDict("functions");
        const FoamDict* s = f ? f->subDict("sample") : nullptr;
        check(s != nullptr, "vacuity guard: the sub-dictionary parsed at all");
        if (s) check(std::fabs(s->scalarOr("timeStart", -1) - 2.0) < 1e-12,
                     "${/endTime} inside #eval #{...#} resolves to the root entry (6/3 = 2)");
    }

    // ---- Leg 2: it really is the REFERENCE, not a constant that happens to match --------------------
    {
        const FoamDict d = parse("endTime 30;\n"
                                 "functions { sample { timeStart #eval #{ 1.0/3.0 * ${/endTime} #}; } }\n");
        const FoamDict* f = d.subDict("functions");
        const FoamDict* s = f ? f->subDict("sample") : nullptr;
        if (s) check(std::fabs(s->scalarOr("timeStart", -1) - 10.0) < 1e-12,
                     "...and it tracks the referenced entry (30/3 = 10)");
    }

    // ---- Leg 3: the unscoped spellings still work, and the brace form of #eval ----------------------
    {
        const FoamDict d = parse("R 0.5;\n"
                                 "a #eval{ 2*$R };\n"
                                 "b #eval{ 2*${R} };\n"
                                 "c #eval{ 2*${/R} };\n"
                                 "e #eval{ $R*sin(0.5*pi()) };\n");
        check(std::fabs(d.scalarOr("a", -1) - 1.0) < 1e-12, "$name still expands");
        check(std::fabs(d.scalarOr("b", -1) - 1.0) < 1e-12, "${name} still expands");
        check(std::fabs(d.scalarOr("c", -1) - 1.0) < 1e-12, "${/name} expands to the same thing");
        check(std::fabs(d.scalarOr("e", -1) - 0.5) < 1e-12, "the function set still evaluates (sin, pi)");
    }

    // ---- Leg 4: a scope form brae does NOT resolve must stay unresolved -----------------------------
    // The danger is not failing; it is succeeding with the wrong entry. `${../R}` names R in the PARENT
    // scope, which brae's flat map cannot distinguish from any other R -- so it must refuse rather than
    // hand back the value of whichever R it happens to hold.
    {
        bool threw = false;
        try
        {
            const FoamDict d = parse("R 0.5;\nouter { R 9; inner { a #eval{ 2*${../R} }; } }\n");
            const FoamDict* o = d.subDict("outer");
            const FoamDict* i = o ? o->subDict("inner") : nullptr;
            if (i) (void)i->scalarOr("a", -1);
        }
        catch (const std::exception&) { threw = true; }
        check(threw, "refusal: a parent-relative ${../name} is not silently resolved");
    }

    std::printf(failures ? "== FAILED (%d) ==\n" : "== PASSED ==\n", failures);
    return failures ? 1 : 0;
}
