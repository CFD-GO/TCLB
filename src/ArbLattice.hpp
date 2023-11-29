#ifndef ARBLATTICE_HPP
#define ARBLATTICE_HPP

#include <cstdint>
#include <map>
#include <string>

#include "ArbConnectivity.hpp"
#include "CudaUtils.hpp"
#include "LatticeBase.hpp"
#include "Lists.h"
#include "Region.h"
#include "pugixml.hpp"
#include "utils.h"

// Note on node numbering: We have 2 indexing schemes - a global and a local one.
// Globally, nodes are numbered such that consecutive ranks own monotonically increasing intervals, (e.g. rank 0 owns nodes [0, n_0), rank 1 owns nodes [n_0, n_1), etc.) This is required for ParMETIS, but it is also a convenient scheme in general.
//     When communicating between ranks, only this numbering is used.
// Locally, nodes are permuted such that:
//     - Owned boundary nodes (i.e. nodes which have ghost nodes among their neighbors) occupy the initial B indices, starting at 0. Their numbering within [0, B) is arbitrary w.r.t. the global scheme
//     - Owned interior nodes occupy the indices [B, B + I), where I is the number of interior nodes. Again, their order within that interval is arbitrary
//     - Ghost nodes (i.e. nodes neighboring owned nodes, but not owned by the current rank) occupy indices [B + I, B + I + G), where G is the number of ghost nodes. **Their numbering within that interval corresponds to the global numbering scheme (important for optimal packing)**
//     - There is at least one padding index B + I + G, corresponding to the row where dummy values (NaNs) will be stored for when a nonexistent neighbor is accessed, more padding may be present to promote coalesced memory access
// Note that the GPU is only aware of the local numbering scheme, which uses 32b indexing, saving memory bandwidth. Local indexing can additionally put "similar" nodes next to each other, minimizing branching and promoting coalesced memory access.
// Naming: GID == global ID; LID == local ID

class ArbLattice : public LatticeBase {
   private:
    struct SizeInfo {
        size_t border_nodes;     /// Number of border nodes, i.e., nodes which have a ghost neighbor
        size_t snaps;            /// Number of snaps to hold
        size_t neighbors_pitch;  /// B + I + padding
        size_t coords_pitch;     /// B + I + padding (should be the same as neighbors_pitch, but let's be extra safe since they come from separate pitched allocation calls)
        size_t snaps_pitch;      /// B + I + G + 1 + padding
    };

    ArbLatticeConnectivity connect;                         /// Lattice connectivity info
    std::vector<long> global_node_dist;                     /// Node distribution (in the ParMETIS sense), describing the GID node intervals owned by each process (identical in all ranks)
    std::vector<long> ghost_nodes;                          /// Sorted GIDs of ghost nodes
    SizeInfo sizes{};                                       /// Sizes of various data structures/allocations
    MPI_Comm comm;                                          /// Communicator associated with the lattice
    std::vector<unsigned> local_permutation;                /// The permutation of owned nodes w.r.t. the global indexing scheme, see comment at the top
    lbRegion local_bounding_box;                            /// The bounding box of the local region (if the arbitrary lattice is a subset of a full Cartesian lattice, we can use this for some optimizations)
    std::unordered_map<std::string, int> label_to_ind_map;  /// Label string to unique ID
    CudaUniquePtr<unsigned> neighbors_device;               /// Device allocation of the neighbor table: (B + I) x Q
    CudaUniquePtr<real_t> coords_device;                    /// Device allocation of node coordinates: (B + I) x 3
    CudaUniquePtr<storage_t> snaps_device;                  /// Device allocation of snaps: (B + I + G + 1) x NF x num_snaps
    CudaUniquePtr<flag_t> node_types_device;                /// Device allocation of node type array: (B + I)
    std::pmr::vector<flag_t> node_types_host;               /// Host (pinned) allocation of node type array: (B + I)
    pugi::xml_node initialized_from;                        /// XML node from which this node was initialized - avoid reinitialization if called multiple times with the same arguments

   public:
    static constexpr size_t Q = Model_m::Q;    /// Stencil size
    static constexpr size_t NF = Model_m::NF;  /// Number of fields

    ArbLattice(size_t num_snaps_, const UnitEnv& units_, const std::map<std::string, int>& setting_zones, pugi::xml_node arb_node, MPI_Comm comm_);
    int reinitialize(size_t num_snaps_, const std::map<std::string, int>& setting_zones, pugi::xml_node arb_node);  /// Init if passed args differ from those passed at construction or the last call to reinitialize (avoid duplicating work)

    size_t getLocalSize() const final { return connect.chunk_end - connect.chunk_begin; }
    size_t getGlobalSize() const final { return connect.num_nodes_global; }

   private:
    struct NodeTypeBrush {
        std::function<bool(Span<const ArbLatticeConnectivity::ZoneIndex>, std::array<double, 3>)> pred;  /// Predicate to determine whether the brush is applicable to this node, takes the labels and coordinates of the node
        flag_t mask, value;                                                                              /// Mask and value of the brush
    };

    void initLatticeDerived() final;                                                 /// Initialization by model/init handlers TODO
    int loadComp(const std::string& filename, const std::string& comp) final;        /// TODO
    int saveComp(const std::string& filename, const std::string& comp) const final;  /// TODO
    int loadPrimal(const std::string& filename, int snap_ind) final;                 /// TODO
    void savePrimal(const std::string& filename, int snap_ind) const final;          /// TODO
#ifdef ADJOINT
    int loadAdj(const std::string& filename, int asnap_ind) final;         /// TODO
    void saveAdj(const std::string& filename, int asnap_ind) const final;  /// TODO
#endif
    void clearAdjoint() final;                                  /// TODO
    void IterationPrimal(int, int, int) final;                  /// TODO
    void IterationAdjoint(int, int, int, int, int) final;       /// TODO
    void IterationOptimization(int, int, int, int, int) final;  /// TODO
    void RunAction(int, int, int, int) final;                   /// TODO

    void initialize(size_t num_snaps_, const std::map<std::string, int>& setting_zones, pugi::xml_node arb_node);                  /// Init based on args
    void readFromCxn(const std::string& cxn_path);                                                                                 /// Read the lattice info from a .cxn file
    void partition();                                                                                                              /// Repartition the lattice, if ParMETIS is not present this is a noop
    void computeLocalPermutation();                                                                                                /// Compute the local permutation, see comment at the top
    void computeGhostNodes();                                                                                                      /// Retrieve GIDs of ghost nodes from the connectivity info structure
    void allocDeviceMemory();                                                                                                      /// Allocate required device memory
    std::vector<NodeTypeBrush> parseBrushFromXml(pugi::xml_node arb_node, const std::map<std::string, int>& setting_zones) const;  /// Parse the arbitrary lattice XML to determine the brush sequence to be applied to each node
    void computeNodeTypesOnHost(pugi::xml_node arb_node, const std::map<std::string, int>& setting_zones);                         /// Compute the node types to be stored on the device
    std::pmr::vector<real_t> computeCoords() const;                                                                                /// Compute the coordinates 2D array to be stored on the device
    std::pmr::vector<unsigned> computeNeighbors() const;                                                                           /// Compute the neighbors 2D array to be stored on the device
    void initDeviceData(pugi::xml_node arb_node, const std::map<std::string, int>& setting_zones);                                 /// Initialize data residing in device memory
    int fullLatticePos(double pos) const;                                                                                          /// Compute the position (in terms of lattice offsets) of a node, assuming the arbitrary lattice is a subset of a Cartesian lattice
    lbRegion getLocalBoundingBox() const;                                                                                          /// Compute local bounding box, assuming the arbitrary lattice is a subset of a Cartesian lattice
};

#endif  // ARBLATTICE_HPP
