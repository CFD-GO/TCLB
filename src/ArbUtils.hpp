#ifndef ARBUTILS_HPP
#define ARBUTILS_HPP

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <unordered_map>
#include <vector>

template <typename T>
class Span {
   public:
    using iterator_t = T*;
    using const_iterator_t = const T*;
    using size_type = size_t;

    Span() = default;
    Span(T* data, size_type size) : data_(data), size_(size) {}
    Span(T* first, T* last) : data_(first), size_(static_cast<size_type>(std::distance(first, last))) {}

    iterator_t begin() { return data_; }
    iterator_t end() { return std::next(data_, size_); }
    const_iterator_t begin() const { return data_; }
    const_iterator_t end() const { return std::next(data_, size_); }
    size_type size() const { return size_; }
    T& front() { return *data_; }
    T& back() { return *std::next(data_, size_ - 1); }
    const T& front() const { return *data_; }
    const T& back() const { return *std::next(data_, size_ - 1); }

   private:
    T* data_ = nullptr;
    size_t size_ = 0;
};

template <typename Vert, typename Size = size_t>
class CrsGraph {
   public:
    using vertex_type = Vert;
    using size_type = Size;

    CrsGraph(Span<size_type> row_sizes) : n_rows_(row_sizes.size()), row_offs_(std::make_unique<size_type[]>(row_sizes.size() + 1)) {
        row_offs_[0] = 0;
        std::inclusive_scan(row_sizes.begin(), row_sizes.end(), std::next(row_offs_.get()));
        cols_ = std::make_unique<vertex_type[]>(row_offs_[n_rows_]);
    }

    size_t numRows() const { return n_rows_; }
    size_t numEntries() const { return row_offs_[n_rows_]; }
    Span<vertex_type> getRow(size_type row) { return {std::next(cols_.get(), row_offs_[row]), std::next(cols_.get(), row_offs_[row + 1])}; }
    Span<const vertex_type> getRow(size_type row) const { return {std::next(cols_.get(), row_offs_[row]), std::next(cols_.get(), row_offs_[row + 1])}; }

   private:
    size_t n_rows_;
    std::unique_ptr<size_type[]> row_offs_;
    std::unique_ptr<vertex_type[]> cols_;
};

template <size_t Q>
struct NodeEntry {
    std::array<double, 3> coords;
    std::array<size_t, Q> nbrs;
    size_t zone_ind;
};

/// Read an arbitrary lattice file
/// \tparam Q - [in] Number of required directions
/// \param path - [in] Path to the arbitrary lattice file
/// \param q_required - [in] Required offset (streaming) directions
/// \param zone_map - [in] Map from zone names to their indices
/// \param nodes - [out] Vector of node data read from the file
/// \return - error code
template <size_t Q>
int readArbLattice(const std::filesystem::path& path, const std::array<std::array<int, 3>, Q>& q_required, const std::unordered_map<std::string, size_t>& zone_map, std::vector<NodeEntry<Q>>& nodes) {
    std::fstream file(path, std::ios_base::in);
#define CHECK_ARB_READ_ERR                                            \
    if (!file) {                                                      \
        std::cerr << "I/O failure while reading arbitrary lattice\n"; \
        return EXIT_FAILURE;                                          \
    }
    CHECK_ARB_READ_ERR;

    /// Header ///
    // Total size of the lattice
    std::string word;
    file >> word;
    CHECK_ARB_READ_ERR;
    if (word != "LATTICESIZE") {
        std::cerr << "Unexpected entry: " << word << '\n';
        return EXIT_FAILURE;
    }
    size_t lat_sz{};
    file >> lat_sz;
    CHECK_ARB_READ_ERR;

    // Offset directions
    file >> word;
    CHECK_ARB_READ_ERR;
    if (word != "OFFSET_DIRECTIONS") {
        std::cerr << "Unexpected entry: " << word << '\n';
        return EXIT_FAILURE;
    }
    size_t n_q_provided{};
    file >> n_q_provided;
    CHECK_ARB_READ_ERR;
    std::vector<std::array<int, 3>> q_provided;
    q_provided.reserve(n_q_provided);
    for (size_t i = 0; i != n_q_provided; ++i) {
        std::array<int, 3> dirs{};
        file >> dirs[0] >> dirs[1] >> dirs[2];
        q_provided.push_back(dirs);
    }
    CHECK_ARB_READ_ERR;

    // Check all required dirs are present, construct the lookup table
    std::array<size_t, Q> req_prov_perm{};  // Lookup table between provided and required dirs
    size_t i = 0;
    for (const auto& req : q_required) {
        const auto prov_it = std::find(q_provided.cbegin(), q_provided.cend(), req);
        if (prov_it == q_provided.cend()) {
            std::cerr << "The arbitrary lattice file does not provide the required direction: [" << req[0] << ", " << req[1] << ", " << req[2] << "]\n";
            return EXIT_FAILURE;
        }
        req_prov_perm[i++] = std::distance(q_provided.cbegin(), prov_it);
    }

    /// Node data ///
    file >> word;
    CHECK_ARB_READ_ERR;
    if (word != "NODES") {
        std::cerr << "Unexpected entry: " << word << '\n';
        return EXIT_FAILURE;
    }

    nodes.clear();
    nodes.reserve(lat_sz);
    std::string zone_name;
    std::vector<size_t> nbrs_in_file;
    nbrs_in_file.resize(n_q_provided);

    // Parse node info
    for (size_t i = 0; i != lat_sz; ++i) {
        size_t node_ind;
        std::array<double, 3> coords;
        auto& [cx, cy, cz] = coords;
        int tag;

        // Read
        file >> node_ind >> cx >> cy >> cz;
        for (size_t& nbr : nbrs_in_file) file >> nbr;
        file >> tag >> zone_name;
        CHECK_ARB_READ_ERR;

        // TODO: for now, we read one contiguous file (MPI stuff to follow)
        if (node_ind != i) {
            std::cerr << "Node indices must form a consecutive interval\n";
            return EXIT_FAILURE;
        }

        // Map read dirs to local ones (i.e. those from q_required)
        std::array<size_t, Q> local_dirs;
        size_t j = 0;
        for (size_t perm : req_prov_perm) local_dirs[j++] = nbrs_in_file[perm];

        const auto zone_it = zone_map.find(zone_name);
        if (zone_it == zone_map.end()) {
            std::cerr << "Encountered unknown zone: " << zone_name << "\n";
            return EXIT_FAILURE;
        }

        nodes.push_back(NodeEntry<Q>{coords, local_dirs, zone_it->second});
    }

    return EXIT_SUCCESS;

#undef CHECK_ARB_READ_ERR
}

#endif  // ARBUTILS_HPP
