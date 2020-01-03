#ifndef LISTS_H
#define LISTS_H
#include <string>

#define LIST_INVALID -1

class SomethingList {
    virtual int IdxFromString (const std::string& str) const;
    virtual std::string StringFromIdx (int i) const;
    virtual const char* CStringFromIdx (int i) const;
    virtual int size () const;
};

class SettingsList : public SomethingList {
    virtual int IdxFromString (const std::string& str) const;
    virtual const char* CStringFromIdx (int i) const;
    virtual int size () const;
};

#endif
