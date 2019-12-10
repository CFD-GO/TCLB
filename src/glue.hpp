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
    std::string begin;
    std::string end;
    bool empty;
public:
    inline Glue() {
        s << std::setprecision(14);
        s << std::scientific;
        empty = true;
    }
    inline Glue(std::string sep_, std::string begin_="", std::string end_="") : sep(sep_), begin(begin_), end(end_) {
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
    inline Glue& clear() {
        s.str(std::string());
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
    inline const std::string& str () {
        val = begin + s.str() + end;
        return val;
    }
    inline const char* c_str () {
        return this->str().c_str();
    }
    inline operator const char* () {
        return this->str().c_str();
    }
};

#endif
