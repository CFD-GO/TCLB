#ifndef UTILS_H
#define UTILS_H

#include <cassert>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <numeric>
#include <set>
#include <string>

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

inline int mkdir_p(const std::string& path, std::filesystem::perms perms = std::filesystem::perms::owner_all | std::filesystem::perms::group_all | std::filesystem::perms::others_read | std::filesystem::perms::others_exec) {
    try {
        const auto fs_path = std::filesystem::path(path).parent_path();
        std::filesystem::create_directories(fs_path);
        std::filesystem::permissions(fs_path, perms);
        return EXIT_SUCCESS;
    } catch (...) {
        debug1("Failed to create %s\n", path.c_str());
        return EXIT_FAILURE;
    }
}

inline FILE* fopen_gz(const char* filename, const char* mode) {
    bool gzip = false;
    int len = strlen(filename);
    if (len > 3) {
        if (strcmp(&filename[len - 3], ".gz") == 0) { gzip = true; }
    }
    if (gzip) {
        warning("Opening a gzip file: %s (%s)\n", filename, mode);
        if (strcmp(mode, "r") == 0) {
            char cmd[STRING_LEN * 2];
            if (access(filename, R_OK)) return NULL;
            sprintf(cmd, "gzip -d <%s", filename);
            return popen(cmd, "r");
        } else if (strcmp(mode, "rb") == 0) {
            char cmd[STRING_LEN * 2];
            if (access(filename, R_OK)) return NULL;
            sprintf(cmd, "gzip -d <%s", filename);
            return popen(cmd, "r");
        } else if (strcmp(mode, "w") == 0) {
            char cmd[STRING_LEN * 2];
            sprintf(cmd, "gzip >%s", filename);
            return popen(cmd, "w");
        } else if (strcmp(mode, "a") == 0) {
            char cmd[STRING_LEN * 2];
            sprintf(cmd, "gzip >>%s", filename);
            return popen(cmd, "w");
        } else {
            ERROR("Unknown mode for gzip file: fopen_gz('%s','%s')\n", filename, mode);
            return NULL;
        }
    } else {
        return fopen(filename, mode);
    }
    return NULL;
}

/// C++17 idiom for grouping together different callables of different types into a single overload set, usually used to visit a variant
template <typename... Fun>
struct OverloadSet : public Fun... {
    using Fun::operator()...;
};
template <typename... Fun>
OverloadSet(Fun...) -> OverloadSet<Fun...>;

/// Small utility for static assertions within `if constexpr` statements
template<typename T>
inline constexpr bool always_false = false;

#endif
