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

class SettingsListBase : public ListBase {
public:
    virtual int DerivedFromIdx(int i) const = 0;
};

class SettingsList : public SettingsListBase {
public:
    virtual int DerivedFromIdx(int i) const;
    virtual double DerivedValueFromIdx(int i, double val) const;
    virtual int IdxFromString (const std::string& str) const;
    virtual const char* CStringFromIdx (int i) const;
    virtual int size () const;
};

class ZoneSettingsListBase : public ListBase {
public:
};

class ZoneSettingsList : public ZoneSettingsListBase {
public:
    virtual int IdxFromString (const std::string& str) const;
    virtual const char* CStringFromIdx (int i) const;
    virtual int size () const;
};

#endif
