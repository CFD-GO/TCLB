#ifndef THINGS_H
#define THINGS_H
#include <vector>
#include <string>

#define INVALID_ID -1

template <class T>
class Things : public std::vector<T> {
    static T& invalid() {
        static T inv;
        return inv;
    }
public:
    const T& by_name(const std::string& str) const {
        for (const T& i : *this) if (i.name == str) return i;
        return invalid();
    }
    const T& by_id(const int& id) const {
        for (const T& i : *this) if (i.id == id) return i;
        return invalid();
    }
    using std::vector<T>::vector;
};        

struct Thing {
    int id;
    std::string name;
    inline Thing() : id(INVALID_ID), name("invalid") {};
    inline Thing(const int& id_, const std::string& name_) : id(id_), name(name_) {}
    inline bool valid() const { return id != INVALID_ID; }
    inline operator bool () const { return valid(); }
};

struct UnitThing : Thing {
    std::string unit;
    inline UnitThing() : unit("invalid") {};
    inline UnitThing(const int& id_, const std::string& name_, const std::string& unit_) : Thing(id_,name_), unit(unit_) {}
};

#endif
