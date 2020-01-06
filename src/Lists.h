#ifndef LISTS_H
#define LISTS_H
#include <string>

#define LIST_INVALID -1

class ListBase {
public: 
    virtual int IdxFromString (const std::string& str) const;
    virtual std::string StringFromIdx (int i) const;
    virtual const char* CStringFromIdx (int i) const;
    virtual int size () const;
};

class UnitListBase : public ListBase {
public:
    virtual std::string UnitFromIdx(int i) const;
};

class SettingsListBase : public UnitListBase {
public:
    virtual int DerivedFromIdx(int i) const = 0;
    virtual double DerivedValueFromIdx(int i, double val) const = 0;
};

class SettingsList : public SettingsListBase {
public:
    virtual int DerivedFromIdx(int i) const;
    virtual double DerivedValueFromIdx(int i, double val) const;
    virtual std::string UnitFromIdx(int i) const;
    virtual int IdxFromString (const std::string& str) const;
    virtual const char* CStringFromIdx (int i) const;
    virtual int size () const;
};

class ZoneSettingsListBase : public UnitListBase {
public:
};

class ZoneSettingsList : public ZoneSettingsListBase {
public:
    virtual std::string UnitFromIdx(int i) const;
    virtual int IdxFromString (const std::string& str) const;
    virtual const char* CStringFromIdx (int i) const;
    virtual int size () const;
};

class QuantitiesListBase : public UnitListBase {
public:
    virtual bool IsVectorFromIdx(int i) const = 0;
    virtual bool IsAdjointFromIdx(int i) const = 0;
};

class QuantitiesList : public QuantitiesListBase {
public:
    virtual bool IsVectorFromIdx(int i) const;
    virtual bool IsAdjointFromIdx(int i) const;
    virtual std::string UnitFromIdx(int i) const;
    virtual int IdxFromString (const std::string& str) const;
    virtual const char* CStringFromIdx (int i) const;
    virtual int size () const;
};


#endif
