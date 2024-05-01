#ifndef UTILS_H
#define UTILS_H

#include <cassert>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <set>
#include <string>
#include <sys/stat.h>

#include "Global.h"
#include "UtilTypes.hpp"

namespace detail {
template <typename T>
T maybeString(const T& t) {
    return t;
}
inline const char* maybeString(const std::string& str) {
    return str.c_str();
}
}  // namespace detail

template <class... Args>
std::string formatAsString(const char* format, Args&&... args) {
    if constexpr (sizeof...(Args) == 0) {
        return format;
    } else {
        const int n_chars = std::snprintf(nullptr, 0, format, detail::maybeString(args)...);
        assert(n_chars >= 0);
        std::string retval;
        retval.resize(static_cast<typename std::string::size_type>(n_chars + 1));
        std::sprintf(&retval[0], format, detail::maybeString(args)...);
        return retval;
    }
}

inline int myround(double v) {
    if (v > 0) return v + 0.5;
    return v - 0.5;
}

class name_set {
    std::set<std::string> myset;

   public:
    bool all;
    typedef std::set<std::string>::iterator iterator;
    void add_from_string(std::string in, char separator) {
        if (in == "all") {
            all = true;
            return;
        }
        std::string::iterator ts, curr;
        ts = curr = in.begin();
        for (; curr <= in.end(); curr++) {
            if ((curr == in.end() || *curr == separator) && curr > ts) myset.insert(std::string(ts, curr));
            if (curr == in.end()) break;
            if (*curr == separator) ts = curr + 1;
        }
    }
    name_set(char* str) {
        all = false;
        add_from_string(str, ',');
    }
    name_set() { all = false; }
    bool in(std::string what) const {
        if (all) return true;
        return myset.count(what) > 0;
    }
    bool explicitlyIn(std::string what) const { return myset.count(what) > 0; }
    int size() { return myset.size(); }
    std::set<std::string>::iterator begin() { return myset.begin(); }
    std::set<std::string>::iterator end() { return myset.end(); }
};

std::string path_stripext(const std::string& str);
std::string path_filename(const std::string& str);

int mkdir_p_c(const char* file_path_, mode_t mode = 0775);
inline int mkdir_p(std::string file_path, mode_t mode = 0775) {
    return mkdir_p_c(file_path.c_str(), mode);
}

void fprintB64(FILE* f, const void* tab, size_t len);

FILE* fopen_gz(const char* filename, const char* mode);

/// C++17 idiom for grouping together different callables of different types into a single overload set, usually used to visit a variant
template <typename... Fun>
struct OverloadSet : public Fun... {
    using Fun::operator()...;
};
template <typename... Fun>
OverloadSet(Fun...) -> OverloadSet<Fun...>;

/// Small utility for static assertions within `if constexpr` statements
template <typename T>
inline constexpr bool always_false = false;

#endif
