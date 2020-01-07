#ifndef LISTS_H
#define LISTS_H
#include <string>
#include <vector>
#include "Consts.h"

#define LIST_INVALID -1

template <class T>
typename T::const_iterator FindByName(const T& cont, const std::string& str) {
    for (typename T::const_iterator it = cont.begin(); it != cont.end(); it++) {
        if (it->name == str) return it;
    }
    return cont.end();
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

struct NodeTypeFlag : Thing {
    flag_t flag;
    int group_id;
};

struct NodeTypeGroupFlag : Thing {
    flag_t flag;
    int shift;
};

class ModelBase {
    template <class T>
    class Things : public std::vector<T> {
    public:
        typename Things::const_iterator ByName(const std::string& str) const {
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
    typedef Things<NodeTypeFlag> NodeTypeFlags;
    NodeTypeFlags nodetypeflags;
    typedef Things<NodeTypeGroupFlag> NodeTypeGroupFlags;
    NodeTypeGroupFlags nodetypegroupflags;
};

class Model_m : public ModelBase {
public:
    Model_m();
};


#endif
