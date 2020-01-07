#ifndef LISTS_H
#define LISTS_H
#include <string>
#include <vector>

#define LIST_INVALID -1

template <class T>
int FindByName(const T& cont, std::string str) {
    for (typename T::const_iterator it = cont.begin(); it != cont.end(); it++) {
        if (it->name == str) return it->id;
    }
    return LIST_INVALID;
}

typedef double (*DerivedFunction)(double);

struct Thing {
    int id;
    std::string name;
};

struct UnitThing : Thing {
    std::string unit;
};

struct Setting : UnitThing {
    bool isDerived;
    int derivedSetting;
    DerivedFunction derivedValue;
};

struct ZoneSetting : UnitThing {
};

struct Quantity : UnitThing {
    bool isVector;
    bool isAdjoint;
};

class ModelBase {
    template <class T>
    class Things : public std::vector<T> {
    public:
        int ByName(const std::string& str) const {
            return FindByName(*this, str);
        }
    };        
public:
    typedef Things<Setting> Settings;
    Settings settings;
    typedef Things<ZoneSetting> ZoneSettings;
    ZoneSettings zonesettings;
    typedef Things<Quantity> Quantities;
    Quantities quantities;
};

class Model_m : public ModelBase {
public:
    Model_m();
};


#endif
