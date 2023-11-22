#ifndef ARBLATTICE_HPP
#define ARBLATTICE_HPP

#include <cstdint>
#include <map>
#include <string>

#include "ArbConnectivity.hpp"
#include "LatticeBase.hpp"
#include "Lists.h"
#include "Region.h"

// Note on node numbering: We have 2 indexing schemes - a global and a local one.
// Globally, nodes are numbered such that consecutive ranks own monotonically increasing intervals, (e.g. rank 0 owns nodes [0, n_0), rank 1 owns nodes [n_0, n_1), etc.) This is required for ParMETIS, but it is also a convenient scheme in general.
//     When communicating between ranks, only this numbering is used.
// Locally, nodes are permuted such that:
//     - -1 is a valid index, signifying that a neighbor is not present at a given direction
//     - Owned boundary nodes (i.e. nodes which have ghost nodes among their neighbors) occupy the initial B indices, starting at 0. Their numbering within [0, B) is arbitrary w.r.t. the global scheme
//     - Owned interior nodes occupy the indices [B, B + I), where I is the number of interior nodes. Again, their order within that interval is arbitrary
//     - Ghost nodes (i.e. nodes neighboring owned nodes, but not owned by the current rank) occupy indices [B + I, B + I + G), where G is the number of ghost nodes. **Their numbering within that interval corresponds to the global numbering scheme (important for optimal packing)**
// Note that the GPU is only aware of the local numbering scheme, which uses 32b indexing, saving memory bandwidth. Local indexing can additionally put "similar" nodes next to each other, minimizing branching and promoting coalesced memory access.
// Naming: GID == global ID; LID == local ID

class ArbLattice : public LatticeBase {
   private:
    ArbLatticeConnectivity connect;           /// Lattice connectivity info
    std::vector<long> global_node_dist;       /// Node distribution (in the ParMETIS sense), describing the GID node intervals owned by each process (identical in all ranks)
    std::vector<long> ghost_nodes;            /// Sorted GIDs of ghost nodes
    size_t num_border_nodes;                  /// Number of border nodes, i.e., nodes which have a ghost neighbor
    lbRegion local_bounding_box;              /// The bounding box of the local region (the arbitrary lattice is a subset of a full Cartesian lattice, we can use this for some optimizations)
    size_t num_snaps;                         /// Number of snaps to hold
    MPI_Comm comm;                            /// Communicator associated with the lattice
    std::vector<unsigned> local_permutation;  /// The permutation of owned nodes w.r.t. the global indexing scheme, see comment at the top

   public:
    static constexpr size_t Q = Model_m::offset_directions.size();

    ArbLattice(size_t num_snaps_, const UnitEnv& units_, const std::map<std::string, int>& zone_map, const std::string& cxn_path, MPI_Comm comm_);

    size_t getLocalSize() const final { return connect.chunk_end - connect.chunk_begin; }
    size_t getGlobalSize() const final { return connect.num_nodes_global; }

    void Iterate(int, int) final {}
    void IterateTill(int, int) final {}

   private:
    std::unordered_map<std::string, int> makeGroupZoneMap(const std::map<std::string, int>& zone_map) const;  /// Translate the groups and zone into a single indexing scheme for the purposes of uniquely mapping read IDs from the .cxn file
    void readFromCxn(const std::map<std::string, int>& zone_map, const std::string& cxn_path);                /// Read the lattice info from a .cxn file
    void partition();                                                                                         /// Repartition the lattice, if ParMETIS is not present this is a noop
    void computeLocalPermutation();                                                                           /// Compute the local permutation, see comment at the top
    void computeGhostNodes();                                                                                 /// Retreive GIDs of ghost nodes from the connectivity info structure
    int fullLatticePos(double pos) const;                                                                     /// Compute the position (in terms of lattice offsets) of a node, assuming the arbitrary lattice is a subset of a Cartesian lattice
    lbRegion getLocalBoundingBox() const;                                                                     /// Compute local bounding box, assuming the arbitrary lattice is a subset of a Cartesian lattice
};

#endif  // ARBLATTICE_HPP
