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

/// Utility abstraction for a dynamically-sized contiguous range of elements of type T
template <typename T>
class Span {
   public:
    Span() = default;
    Span(T* data, size_t size) : data_(data), size_(size) {}
    Span(T* first, T* last) : data_(first), size_(static_cast<size_t>(std::distance(first, last))) {}
    template <typename Iterator, std::enable_if_t<!std::is_pointer_v<Iterator>, bool> = true>
    Span(Iterator first, Iterator last) : Span(std::addressof(*first), std::addressof(*last)) {}  // Be careful not to call this with a non-contiguous iterator pair, std::contiguous_iterator is locked behind C++20
    template <typename T_other, std::enable_if_t<std::is_same_v<T, typename std::add_const_t<T>> && std::is_same_v<T_other, typename std::remove_const_t<T>>, bool> = true>
    Span(Span<T_other> other) : Span(other.begin(), other.end()) {}  // Converting constructor: Span<T> -> Span<const T>

    T* begin() const { return data_; }
    T* end() const { return std::next(data_, size_); }
    size_t size() const { return size_; }
    T& front() const { return *data_; }
    T& back() const { return *std::next(data_, size_ - 1); }
    T& operator[](size_t i) const { return data_[i]; }
    T* data() const { return data_; }

   private:
    T* data_ = nullptr;
    size_t size_ = 0;
};
template <typename Iterator>
Span(Iterator, Iterator) -> Span<typename std::remove_reference_t<decltype(*std::declval<Iterator>())>>;  // std::iterator_traits ignores constness, hence the hand-rolled type inspection

/// Compressed-Row-Sparse format graph utility
template <typename Vert, typename Size = size_t>
class CrsGraph {
   public:
    using vertex_type = Vert;
    using size_type = Size;

    CrsGraph() = default;
    CrsGraph(Span<const size_type> row_sizes) : n_rows_(static_cast<size_type>(row_sizes.size())), row_offs_(std::make_unique<size_type[]>(row_sizes.size() + 1)) {
        row_offs_[0] = 0;
        std::inclusive_scan(row_sizes.begin(), row_sizes.end(), std::next(row_offs_.get()));
        cols_ = std::make_unique<vertex_type[]>(row_offs_[n_rows_]);
    }

    size_type numRows() const { return n_rows_; }
    size_type numEntries() const { return row_offs_[n_rows_]; }
    Span<vertex_type> getRow(size_type row) { return {std::next(cols_.get(), row_offs_[row]), std::next(cols_.get(), row_offs_[row + 1])}; }
    Span<const vertex_type> getRow(size_type row) const { return {std::next(cols_.get(), row_offs_[row]), std::next(cols_.get(), row_offs_[row + 1])}; }

    /// Danger zone ///
    Span<size_type> getRawOffsets() const { return {row_offs_.get(), static_cast<size_t>(n_rows_ + 1)}; }
    Span<vertex_type> getRawEntries() const { return {cols_.get(), static_cast<size_t>(numEntries())}; }
    ///////////////////

   private:
    size_type n_rows_ = 0;
    std::unique_ptr<size_type[]> row_offs_;
    std::unique_ptr<vertex_type[]> cols_;
};
template <typename Vertex>
CrsGraph(Span<Vertex>) -> CrsGraph<typename std::remove_const_t<Vertex>, typename std::remove_const_t<Vertex>>;

#endif
