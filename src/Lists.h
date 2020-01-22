#ifndef LISTS_H
#define LISTS_H
#include <string>
#include <vector>
#include "Consts.h"
#include "types.h"
#include "mpi.h"

#define LIST_INVALID -1

template <class T>
typename T::const_iterator FindByName(const T& cont, const std::string& str) {
    for (typename T::const_iterator it = cont.begin(); it != cont.end(); it++) {
        if (it->name == str) return it;
    }
    return cont.end();
}

template <class T>
typename T::const_iterator FindById(const T& cont, const int& id) {
    for (typename T::const_iterator it = cont.begin(); it != cont.end(); it++) {
        if (it->id == id) return it;
    }
    return cont.end();
}

typedef double (*DerivedFunction)(double);
typedef void (*ObjectiveFunction)(double*, double*, double*);

class ModelBase {
    template <class T>
    class Things : public std::vector<T> {
    public:
        typename Things::const_iterator ByName(const std::string& str) const {
            return FindByName(*this, str);
        }
        typename Things::const_iterator ById(const int& id) const {
            return FindById(*this, id);
        }
    };        

    struct Thing {
        int id;
        std::string name;
    };

    struct UnitThing : Thing {
        std::string unit;
    };

public:

    struct Setting : UnitThing {
        bool isDerived;
        int derivedSetting;
        DerivedFunction derivedValue;
        std::string defaultValue;
    };

    struct ZoneSetting : UnitThing {
        std::string defaultValue;
    };

    struct Quantity : UnitThing {
        bool isVector;
        bool isAdjoint;
    };

    struct NodeTypeFlag : Thing {
        big_flag_t flag;
        big_flag_t group_flag;
        int group_id;
    };

    struct NodeTypeGroupFlag : Thing {
        big_flag_t flag;
        int shift;
        int max;
        int capacity;
        int bits;
        bool isSave;
    };

    struct Global : UnitThing {
        bool isAdjoint;
        MPI_Op operation;
        int inObjId;
    };

    struct Option : Thing {
        bool isActive;
    };

    struct Scale : UnitThing {
    };

    struct Field : Thing {
        bool isAdjoint;
        bool isParameter;
        bool isAverage;
        std::string adjointName;
        std::string tangentName;
        int accessArea;
        bool simpleAccess;
        std::string niceName;
    };

    struct Action : Thing {
        std::vector<int> stages;
    };

    struct Stage : Thing {
        bool isAdjoint;
        bool isParticle;
        std::string mainFun;
    };

    struct Objective : Thing {
        ObjectiveFunction fun;
    };

    std::string name;
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
    typedef Things<Global> Globals;
    Globals globals;
    typedef Things<Option> Options;
    Options options;
    typedef Things<Scale> Scales;
    Scales scales;
    typedef Things<Field> Fields;
    Fields fields;
    typedef Things<Action> Actions;
    Actions actions;
    typedef Things<Stage> Stages;
    Stages stages;
    NodeTypeGroupFlag settingzones;
    typedef Things<Objective> Objectives;
    Objectives objectives;
};

class Model_m : public ModelBase {
public:
    Model_m();
};


#endif
