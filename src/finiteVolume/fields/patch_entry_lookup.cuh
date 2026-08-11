#pragma once
// patch_entry_lookup.cuh -- find the boundaryField entry that applies to a mesh patch, OF's way.
//
// OpenFOAM resolves a boundaryField key against a patch in three steps: exact name, then literal group
// membership, then REGEX against the patch name or any of its group names, with the LAST matching pattern
// winning. Cases lean on this constantly -- `"(?i).*walls"`, `".*"`, `"(inlet|outlet)"`.
//
// buildField (geometric_field.cuh) has always done this correctly. Several other places did NOT: they
// compared `entry.name == patch.name` and silently fell back to a default when a case used a regex key.
// The result was a plausible run with the wrong coefficient:
//
//   - the per-face Prt for alphatWallFunction reverted to the model default 1.0 instead of the wall
//     function's 0.85, i.e. wall alphat and the wall heat flux ~15% low (squareBend* key their alphat
//     wall entry as "(?i).*walls" while the mesh patch is literally `walls`);
//   - turbulent inlet BCs kept their written `value` placeholder instead of the computed inlet value.
//
// Same rule, one implementation, so a regex-keyed case cannot resolve differently depending on which
// piece of code is asking.

#include "foam_dict.cuh"   // compileFoamRegex ((?i) flag support)
#include "fv_patch.cuh"
#include <regex>
#include <string>
#include <vector>

namespace brae {

// The entry that applies to `p`, or nullptr. Entries must expose `.name`; any PatchFieldData<T> does.
template <typename Entry, typename Patch>
inline const Entry* findPatchEntry(const std::vector<Entry>& entries, const Patch& p)
{
    for (const Entry& b : entries)                       // 1. exact name wins outright
        if (b.name == p.name) return &b;

    const Entry* hit = nullptr;                          // 2./3. group, then regex; LAST match wins (OF)
    for (const Entry& b : entries)
    {
        bool match = false;
        for (const std::string& g : p.inGroups)
            if (b.name == g) { match = true; break; }
        if (!match)
        {
            try
            {
                const std::regex re = compileFoamRegex(b.name);
                if (std::regex_match(p.name, re)) match = true;
                else
                    for (const std::string& g : p.inGroups)
                        if (std::regex_match(g, re)) { match = true; break; }
            }
            catch (const std::regex_error&) {}
        }
        if (match) hit = &b;
    }
    return hit;
}

}   // namespace brae
