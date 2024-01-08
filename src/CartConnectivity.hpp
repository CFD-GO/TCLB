#ifndef CARTCONNECTIVITY_HPP
#define CARTCONNECTIVITY_HPP

#include "cross.h"
#include "pinned_allocator.hpp"
#include "Region.h"

/// Information on connectivity of a processor
struct NodeInfo {
    lbRegion region; ///< Local Lattice region
    int side[27];
};

struct CartConnectivity {
    std::vector<NodeInfo> nodes;
    lbRegion global_region;
    int divx, divy, divz; ///< MPI divisions
    int mpi_rank; ///< my MPI rank

    const lbRegion& getLocalRegion() const { return nodes[mpi_rank].region; }
};

#endif  // CARTCONNECTIVITY_HPP
