#ifndef GLUE_H
#define GLUE_H

#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>

class Glue {
private:
    std::stringstream s;
    std::string sep;
    std::string val;
    bool empty;
public:
    inline Glue() {
        s << std::setprecision(14);
        s << std::scientific;
        empty = true;
    }
    inline Glue& operator () (std::string sep_ = "") {
        s.str(std::string());
        sep = sep_;
        empty = true;
        return *this;
    }
    template <class T> inline Glue& operator<< (const std::pair<T*, int>& t) {
        for (int i=0; i<t.second; i++) (*this) << t.first[i];
        return *this;
    }
    template <class T> inline Glue& operator<< (const T& t) {
        if (sep == "" || empty)
            s << t;
        else
            s << sep << t;
        empty = false;
        return *this;
    }
    inline const char* c_str () {
        val = s.str();
        return val.c_str();
    }
    inline operator const char* () {
        return this->c_str();
    }
};

#endif
