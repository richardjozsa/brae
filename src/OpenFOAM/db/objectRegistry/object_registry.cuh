#pragma once
// brae::ObjectRegistry -- OF src/OpenFOAM/db/objectRegistry.
//
// WHY. In OpenFOAM, `class Time : public objectRegistry` (Time.H:74-80), so a functionObject never
// holds a reference to a solver or to a field: it asks the registry by NAME, and it asks LATE --
// scalarTransport.C's transportedField() looks up (or creates) its field on first use, not at
// construction. That indirection is what makes construction order a non-issue there.
//
// brae had no equivalent, and the cost was concrete: ScalarTransportFO captured DeviceSimpleSolver& at
// construction, so Time had to be built AFTER the solver. gpuSimpleFoam builds Time at start-up (so the
// functionObject report survives a later refusal) and its solver 230 lines further on, which made
// scalarTransport impossible to register there at all -- not for any physical reason, purely ordering.
//
// DELIBERATELY NOT OF's objectRegistry. OF's is a HashTable<regIOobject*> that owns I/O, checkIn/checkOut
// and dependency tracking. This one only needs to answer "is X available yet, and where is it" for
// objects the solver has already built, so it is a name -> (void*, type) map with a type check on
// lookup. Nothing is owned: the solver keeps ownership, the registry only points.
//
// lookupObject returns a POINTER, not a reference, and null for absent. OF throws, because in OF an
// absent object at execute() time is a case error. Here absence is the normal early state -- the whole
// point is that a functionObject may be constructed before the thing it needs exists -- so the caller
// checks and defers rather than failing.

#include <map>
#include <type_traits>
#include <string>
#include <typeindex>
#include <typeinfo>

namespace brae {

class ObjectRegistry
{
public:
    // The solver stores what a functionObject may need, as soon as it exists. Re-storing the same name
    // replaces the entry, which is what a restart or a rebuilt solver wants.
    // T may be const-qualified (a solver registers `const std::vector<FvPatch>*` for its patches), so the
    // constness is cast away for storage and reapplied by the caller's lookup type. typeid ignores
    // top-level cv, so `const X` and `X` share one key -- which is what we want: the entry is the same
    // object either way, and only the borrower decides whether it may mutate it.
    template <class T>
    void store(const std::string& name, T* obj)
    {
        objects_[name] = Entry{
            const_cast<void*>(static_cast<const void*>(obj)),
            std::type_index(typeid(typename std::remove_const<T>::type))};
    }

    // OF objectRegistry::foundObject<Type>(name). Type-checked: a name registered as one type is not
    // found as another, so a mismatch is a miss rather than a bad cast.
    template <class T>
    bool foundObject(const std::string& name) const
    {
        const auto it = objects_.find(name);
        return it != objects_.end()
            && it->second.type == std::type_index(typeid(typename std::remove_const<T>::type));
    }

    // OF objectRegistry::lookupObject<Type>(name), returning null instead of throwing -- see the header
    // note: absence is the expected early state here, not an error.
    template <class T>
    T* lookupObject(const std::string& name) const
    {
        const auto it = objects_.find(name);
        if (it == objects_.end()
            || it->second.type != std::type_index(typeid(typename std::remove_const<T>::type)))
            return nullptr;
        return static_cast<T*>(it->second.ptr);
    }

    bool erase(const std::string& name) { return objects_.erase(name) != 0; }
    std::size_t size() const { return objects_.size(); }

private:
    struct Entry
    {
        void*           ptr  = nullptr;
        std::type_index type = std::type_index(typeid(void));
    };
    std::map<std::string, Entry> objects_;
};

}   // namespace brae
