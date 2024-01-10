#ifndef ARBCONNECTIVITY_HPP
#define ARBCONNECTIVITY_HPP

#include <memory>
#include <numeric>
#include <vector>

struct ArbLatticeConnectivity {
    using Index = long;
    using ZoneIndex = unsigned short;

    size_t chunk_begin{}, chunk_end{}, num_nodes_global{}, Q{};
    std::unique_ptr<double[]> coords;
    std::unique_ptr<Index[]> og_index;
    std::unique_ptr<Index[]> nbrs;
    std::unique_ptr<ZoneIndex[]> zones_per_node;
    std::vector<ZoneIndex> zones;
    double grid_size{};

    ArbLatticeConnectivity() = default;
    ArbLatticeConnectivity(size_t chunk_begin_, size_t chunk_end_, size_t num_nodes_global_, size_t Q_)
        : chunk_begin(chunk_begin_),
          chunk_end(chunk_end_),
          num_nodes_global(num_nodes_global_),
          Q(Q_),
          coords(std::make_unique<double[]>(3 * (chunk_end_ - chunk_begin_))),
          og_index(std::make_unique<Index[]>(chunk_end_ - chunk_begin_)),
          nbrs(std::make_unique<Index[]>((chunk_end_ - chunk_begin_) * Q)),
          zones_per_node(std::make_unique<ZoneIndex[]>(chunk_end_ - chunk_begin_)) {
        zones.reserve(getLocalSize());
    }

    void dump(std::string filename) const {
        FILE* f;
        f = fopen(filename.c_str(), "w");
        fprintf(f, "idx_og,idx");
        for (size_t q = 0; q < Q; q++) fprintf(f, ",nbr%ld", q);
        fprintf(f, "\n");
        size_t n = chunk_end - chunk_begin;
        for (size_t lid = 0; lid < n; lid++) {
            fprintf(f, "%ld,%ld", (size_t)og_index[lid], (size_t)lid + chunk_begin);
            for (size_t q = 0; q < Q; q++) fprintf(f, ",%ld", (signed long int)neighbor(q, lid));
            fprintf(f, "\n");
        }
        fclose(f);
    }

    size_t getLocalSize() const { return chunk_end - chunk_begin; }
    bool isGhost(Index nbr) const { return nbr != -1 && (nbr < static_cast<Index>(chunk_begin) || nbr >= static_cast<Index>(chunk_end)); }

    double& coord(size_t dim, size_t local_node_ind) { return coords[local_node_ind + dim * getLocalSize()]; }
    double coord(size_t dim, size_t local_node_ind) const { return coords[local_node_ind + dim * getLocalSize()]; }
    Index& neighbor(size_t q, size_t local_node_ind) { return nbrs[local_node_ind + q * getLocalSize()]; }
    Index neighbor(size_t q, size_t local_node_ind) const { return nbrs[local_node_ind + q * getLocalSize()]; }
};

inline auto computeInitialNodeDist(size_t num_nodes_global, size_t comm_size) -> std::vector<long> {
    const auto chunk_size = num_nodes_global / comm_size;
    const auto div_remainder = num_nodes_global % comm_size;
    auto retval = std::vector<long>{};
    retval.reserve(comm_size + 1);
    retval.push_back(0);
    for (size_t i = 0; i != comm_size; ++i) retval.push_back(static_cast<long>(chunk_size + (i < div_remainder)));
    std::inclusive_scan(retval.begin(), retval.end(), retval.begin());
    return retval;
}

#endif  // ARBCONNECTIVITY_HPP
