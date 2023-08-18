#ifndef LISTS_H
#define LISTS_H
#include <string>
#include <vector>
#include "Consts.h"
#include "types.h"
#include <mpi.h>

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
        using std::vector<T>::vector;
    };        

    struct Thing {
        int id;
        std::string name;
        inline Thing() {};
        inline Thing(const int& id_, const std::string& name_) : id(id_), name(name_) {}
    };

    struct UnitThing : Thing {
        std::string unit;
        inline UnitThing() {};
        inline UnitThing(const int& id_, const std::string& name_, const std::string& unit_) : Thing(id_,name_), unit(unit_) {}
    };

public:

    struct Setting : UnitThing {
        std::string defaultValue;
        bool isDerived;
        int derivedSetting;
        DerivedFunction derivedValue;
        inline Setting() {};
        inline Setting(const int& id_, const std::string& name_, const std::string& unit_, 
                const std::string& defaultValue_, const bool& isDerived_, const int& derivedSetting_, const DerivedFunction& derivedValue_)
            : UnitThing(id_,name_,unit_), defaultValue(defaultValue_), isDerived(isDerived_), derivedSetting(derivedSetting_), derivedValue(derivedValue_) {}
        inline Setting(const int& id_, const std::string& name_, const std::string& unit_, const std::string& defaultValue_)
            : UnitThing(id_,name_,unit_), defaultValue(defaultValue_), isDerived(false) {}
    };

    struct ZoneSetting : UnitThing {
        std::string defaultValue;
        inline ZoneSetting(const int& id_, const std::string& name_, const std::string& unit_, const std::string& defaultValue_)
            : UnitThing(id_,name_,unit_), defaultValue(defaultValue_) {}
    };

    struct Quantity : UnitThing {
        bool isVector;
        bool isAdjoint;
        inline Quantity(const int& id_, const std::string& name_, const std::string& unit_, const bool& isVector_, const bool& isAdjoint_=false)
            : UnitThing(id_,name_,unit_), isVector(isVector_), isAdjoint(isAdjoint_) {}
    };

    struct NodeTypeFlag : Thing {
        big_flag_t flag;
        big_flag_t group_flag;
        int group_id;
        inline NodeTypeFlag(const big_flag_t& flag_, const std::string& name_, const big_flag_t& group_flag_)
            : Thing(flag_,name_), flag(flag_), group_id(group_flag_), group_flag(group_flag_) {}
    };

    struct NodeTypeGroupFlag : Thing {
        big_flag_t flag;
        int shift;
        int max;
        int capacity;
        int bits;
        bool isSave;
        inline NodeTypeGroupFlag() {} // this is kept for default initialization of: NodeTypeGroupFlag settingzones;
        inline NodeTypeGroupFlag(const big_flag_t& flag_, const std::string& name_,
                const int& shift_, const int& max_, const int& capacity_, const int& bits_, const bool& isSave_)
            : Thing(flag_,name_), flag(flag_), shift(shift_), max(max_), capacity(capacity_), bits(bits_), isSave(isSave_) {}
    };

    struct Global : UnitThing {
        // set.inObjId = <?%s v$Index ?> + IN_OBJ_OFFSET;
        bool isAdjoint;
        MPI_Op operation;
        int inObjId;
        // | id                   | name           | unit    | operation | isAdjoint |
        inline Global(const int& id_, const std::string& name_, const std::string& unit_,
                const MPI_Op& operation_, const bool& isAdjoint_=false)
            : UnitThing(id_,name_,unit_), operation(operation_), inObjId(id_ + IN_OBJ_OFFSET), isAdjoint(isAdjoint_) {}
    };

    struct Option : Thing {
        bool isActive;
        inline Option(const int& id_, const std::string& name_, const bool& isActive_)
            : Thing(id_,name_), isActive(isActive_) {}
    };

    struct Scale : UnitThing {
        using UnitThing::UnitThing;
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
        inline Field(const int& id_, const std::string& name_, const std::string& niceName_,
                const bool& isParameter_, const bool& isAverage_, const int& accessArea_, const bool& simpleAccess_)
            : Thing(id_,name_), niceName(niceName_)
            , isParameter(isParameter_), isAverage(isAverage_), accessArea(accessArea_), simpleAccess(simpleAccess_)
            , isAdjoint(false) {}
        inline Field(const int& id_, const std::string& name_, const std::string& niceName_,
                const bool& isParameter_, const bool& isAverage_, const int& accessArea_, const bool& simpleAccess_,
                const bool& isAdjoint_, const std::string& adjointName_, const std::string& tangentName_)
            : Thing(id_,name_), niceName(niceName_)
            , isParameter(isParameter_), isAverage(isAverage_), accessArea(accessArea_), simpleAccess(simpleAccess_)
            , isAdjoint(isAdjoint_), adjointName(adjointName_), tangentName(tangentName_) {}
    };

    struct Action : Thing {
        std::vector<int> stages;
        inline Action(const int& id_, const std::string& name_, const std::vector<int>& stages_)
            : Thing(id_,name_), stages(stages_) {}
    };

    struct Stage : Thing {
        bool isAdjoint;
        bool isParticle;
        std::string mainFun;
        inline Stage(const int& id_, const std::string& name_, const bool& isParticle_, const std::string& mainFun_, const bool& isAdjoint_ = false)
            : Thing(id_,name_), isParticle(isParticle_), isAdjoint(isAdjoint_), mainFun(mainFun_) {}
    };

    struct Objective : Thing {
        ObjectiveFunction fun;
        inline Objective(const int& id_, const std::string& name_, const ObjectiveFunction& fun_)
            : Thing(id_,name_), fun(fun_) {}
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
