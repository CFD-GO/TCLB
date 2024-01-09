#ifndef UTILTYPES_HPP
#define UTILTYPES_HPP

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

#endif  // UTILTYPES_HPP
