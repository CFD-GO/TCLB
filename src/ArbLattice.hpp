#ifndef ARBLATTICE_HPP
#define ARBLATTICE_HPP

#include <cstdint>
#include <map>
#include <string>

#include "ArbConnectivity.hpp"
#include "ArbLatticeLauncher.h"
#include "CudaUtils.hpp"
#include "LatticeBase.hpp"
#include "Lists.h"
#include "Region.h"
#include "pugixml.hpp"
#include "utils.h"

/// Note on node numbering: We have 2 indexing schemes - a global and a local one.
/// Globally, nodes are numbered such that consecutive ranks own monotonically increasing intervals, (e.g. rank 0 owns nodes [0, n_0), rank 1 owns nodes [n_0, n_1), etc.) This is required for ParMETIS, but it is also a convenient scheme in general.
///     When communicating between ranks, only this numbering is used.
/// Locally, nodes are permuted such that:
///     - Owned boundary nodes (i.e. nodes which have ghost nodes among their neighbors) occupy the initial B indices, starting at 0. Their numbering within [0, B) is arbitrary w.r.t. the global scheme
///     - Owned interior nodes occupy the indices [B, B + I), where I is the number of interior nodes. Again, their order within that interval is arbitrary
///     - Ghost nodes (i.e. nodes neighboring owned nodes, but not owned by the current rank) occupy indices [B + I, B + I + G), where G is the number of ghost nodes. **Their numbering within that interval corresponds to the global numbering scheme (important for optimal packing)**
///     - There is at least one padding index B + I + G, corresponding to the row where dummy values (NaNs) will be stored for when a nonexistent neighbor is accessed, more padding may be present to promote coalesced memory access
/// Note that the GPU is only aware of the local numbering scheme, which uses 32b indexing, saving memory bandwidth. Local indexing can additionally put "similar" nodes next to each other, minimizing branching and promoting coalesced memory access.
/// Naming: GID == global ID; LID == local ID

class ArbLattice : public LatticeBase {
    /// Group sizes of various structures together
    struct SizeInfo {
        size_t border_nodes;     /// Number of border nodes, i.e., nodes which have a ghost neighbor
        size_t snaps;            /// Number of snaps to hold, including adjoint snaps (if present)
        size_t neighbors_pitch;  /// B + I + padding
        size_t coords_pitch;     /// B + I + padding (should be the same as neighbors_pitch, but let's be extra safe since they come from separate pitched allocation calls)
        size_t snaps_pitch;      /// B + I + G + 1 + padding
    };

    struct CommManager {
        std::vector<std::pair<int, size_t>> in_nbrs, out_nbrs;      /// Neighbor IDs + how many elements they are sending
        std::pmr::vector<storage_t> recv_buf_host, send_buf_host;   /// Comm buffers - these are sent/received on host | TODO: CUDA + MPI
        CudaUniquePtr<storage_t> recv_buf_device, send_buf_device;  /// Comm buffers - these are packed and unpacked on the device
        CudaUniquePtr<size_t> unpack_inds, pack_inds;               /// Recipes for how to pack/unpack the comm buffers from/into snaps
    };

   public:
    /// Struct for storing arb lattice geometry info for export to VTU (ParaView unstructured grid format). Note that the ordering is permuted, so that solution data can be directly written to the results file
    struct ArbVTUGeom {
        size_t num_cubes, num_verts;        /// Number of cubes (equal to number of nodes) and number of unique cube vertices
        std::unique_ptr<double[]> coords;   /// Vertex coordinates, stored as Aos, verts x 3
        std::unique_ptr<unsigned[]> verts;  /// Vertex indices, stored as Aos, nodes x 8
    };

   private:
    ArbLatticeConnectivity connect;                         /// Lattice connectivity info
    std::vector<long> global_node_dist;                     /// Node distribution (in the ParMETIS sense), describing the GID node intervals owned by each process (identical in all ranks)
    std::vector<long> ghost_nodes;                          /// Sorted GIDs of ghost nodes
    SizeInfo sizes{};                                       /// Sizes of various data structures/allocations
    MPI_Comm comm;                                          /// Communicator associated with the lattice
    CommManager comm_manager;                               /// Object for managing data required for MPI communication of border snap values
    std::vector<unsigned> local_permutation;                /// The permutation of owned nodes w.r.t. the global indexing scheme, see comment at the top
    lbRegion local_bounding_box;                            /// The bounding box of the local region (if the arbitrary lattice is a subset of a full Cartesian lattice, we can use this for some optimizations)
    ArbVTUGeom vtu_geom;                                    /// Pre-computed geometry of the lattice for export to .vtu
    std::unordered_map<std::string, int> label_to_ind_map;  /// Label string to unique ID
    CudaUniquePtr<unsigned> neighbors_device;               /// Device allocation of the neighbor table: (B + I) x Q
    CudaUniquePtr<real_t> coords_device;                    /// Device allocation of node coordinates: (B + I) x 3
    CudaUniquePtr<storage_t> snaps_device;                  /// Device allocation of snaps: (B + I + G + 1) x NF x num_snaps
    CudaUniquePtr<flag_t> node_types_device;                /// Device allocation of node type array: (B + I)
    std::pmr::vector<flag_t> node_types_host;               /// Host (pinned) allocation of node type array: (B + I)
    pugi::xml_node initialized_from;                        /// XML node from which this node was initialized - avoid reinitialization if called multiple times with the same arguments
    std::string debug_name;                                 /// Prefix of debug files. Debug files are dumped if debug_name != ""
   public:
    static constexpr size_t Q = Model_m::Q;    /// Stencil size
    static constexpr size_t NF = Model_m::NF;  /// Number of fields

    ArbLattice(size_t num_snaps_, const UnitEnv& units_, const std::map<std::string, int>& setting_zones, pugi::xml_node arb_node, MPI_Comm comm_);
    ArbLattice(const ArbLattice&) = delete;
    ArbLattice(ArbLattice&&) = delete;
    ArbLattice& operator=(const ArbLattice&) = delete;
    ArbLattice& operator=(ArbLattice&&) = delete;
    virtual ~ArbLattice() = default;

    int reinitialize(size_t num_snaps_, const std::map<std::string, int>& setting_zones, pugi::xml_node arb_node);  /// Init if passed args differ from those passed at construction or the last call to reinitialize (avoid duplicating work)
    size_t getLocalSize() const final { return connect.chunk_end - connect.chunk_begin; }
    size_t getGlobalSize() const final { return connect.num_nodes_global; }


    virtual std::vector<int> shape() const { return {static_cast<int>(getLocalSize())}; };
    virtual std::vector<real_t> getQuantity(const Model::Quantity& q, real_t scale = 1) ;
    virtual std::vector<big_flag_t> getFlags() const;
    virtual std::vector<real_t> getField(const Model::Field& f);
    virtual std::vector<real_t> getFieldAdj(const Model::Field& f);
    virtual std::vector<real_t> getCoord(const Model::Coord& q, real_t scale = 1);

    virtual void setFlags(const std::vector<big_flag_t>& x);
    virtual void setField(const Model::Field& f, const std::vector<real_t>& x);
    virtual void setFieldAdjZero(const Model::Field& f);
    
    const ArbVTUGeom& getVTUGeom() const { return vtu_geom; }
    Span<const flag_t> getNodeTypes() const { return {node_types_host.data(), node_types_host.size()}; }  /// Get host view of node types (permuted)
    const ArbLatticeConnectivity& getConnectivity() const { return connect; }
    const std::vector<unsigned>& getLocalPermutation() const { return local_permutation; }

   protected:
    ArbLatticeLauncher launcher;  /// Launcher responsible for running CUDA kernels on the lattice
    void SetFirstTabs(int tab0, int tab1);
    void setSnapIn(int tab) { launcher.container.snap_in = getSnapPtr(tab); }
    void setSnapOut(int tab) { launcher.container.snap_out = getSnapPtr(tab); }
#ifdef ADJOINT
    void setAdjSnapIn(int tab) { launcher.container.adj_snap_in = getAdjointSnapPtr(tab); }
    void setAdjSnapOut(int tab) { launcher.container.adj_snap_out = getAdjointSnapPtr(tab); }
#endif
    void MPIStream_A();
    void MPIStream_B();

   private:
    struct NodeTypeBrush {
        std::function<bool(Span<const ArbLatticeConnectivity::ZoneIndex>, std::array<double, 3>)> pred;  /// Predicate to determine whether the brush is applicable to this node, takes the labels and coordinates of the node
        flag_t mask, value;                                                                              /// Mask and value of the brush
    };

    storage_t* getSnapPtr(int snap_ind);  /// Get device pointer to the specified snap (somewhere within the total snap allocation)
    const storage_t* getSnapPtr(int snap_ind) const { return const_cast<ArbLattice*>(this)->getSnapPtr(snap_ind); }
#ifdef ADJOINT
    storage_t* getAdjointSnapPtr(int snap_ind);  /// Get device pointer to the specified adjoint snap, snap_ind must be 0 or 1
#endif

    int loadPrimal(const std::string& filename, int snap_ind) final;                 /// TODO
    void savePrimal(const std::string& filename, int snap_ind) const final;          /// TODO
#ifdef ADJOINT
    int loadAdj(const std::string& filename, int asnap_ind) final;         /// TODO
    void saveAdj(const std::string& filename, int asnap_ind) const final;  /// TODO
#endif
    void clearAdjoint() final;  /// TODO

    void initialize(size_t num_snaps_, const std::map<std::string, int>& setting_zones, pugi::xml_node arb_node);                  /// Init based on args
    void readFromCxn(const std::string& cxn_path);                                                                                 /// Read the lattice info from a .cxn file
    void partition();                                                                                                              /// Repartition the lattice, if ParMETIS is not present this is a noop
    std::function<bool(int, int)> makePermCompare(pugi::xml_node arb_node, const std::map<std::string, int>& setting_zones);       /// Make type-erased comparison operator for computing the local permutation, according to the strategy specified in the xml file
    void computeLocalPermutation(pugi::xml_node arb_node, const std::map<std::string, int>& setting_zones);                        /// Compute the local permutation, see comment at the top
    void computeGhostNodes();                                                                                                      /// Retrieve GIDs of ghost nodes from the connectivity info structure
    void allocDeviceMemory();                                                                                                      /// Allocate required device memory
    std::vector<NodeTypeBrush> parseBrushFromXml(pugi::xml_node arb_node, const std::map<std::string, int>& setting_zones) const;  /// Parse the arbitrary lattice XML to determine the brush sequence to be applied to each node
    void computeNodeTypesOnHost(pugi::xml_node arb_node, const std::map<std::string, int>& setting_zones, bool permute);           /// Compute the node types to be stored on the device, `permute` enables better code reuse
    std::pmr::vector<real_t> computeCoords() const;                                                                                /// Compute the coordinates 2D array to be stored on the device
    std::pmr::vector<unsigned> computeNeighbors() const;                                                                           /// Compute the neighbors 2D array to be stored on the device
    void initDeviceData(pugi::xml_node arb_node, const std::map<std::string, int>& setting_zones);                                 /// Initialize data residing in device memory
    void initCommManager();                                                                                                        /// Compute which fields need to be sent to/received from which neighbors
    void initContainer();                                                                                                          /// Initialize the data residing in launcher.container
    int fullLatticePos(double pos) const;                                                                                          /// Compute the position (in terms of lattice offsets) of a node, assuming the arbitrary lattice is a subset of a Cartesian lattice
    lbRegion getLocalBoundingBox() const;                                                                                          /// Compute local bounding box, assuming the arbitrary lattice is a subset of a Cartesian lattice
    ArbVTUGeom makeVTUGeom() const;                                                                                                /// Compute VTU geometry
    void communicateBorder();                                                                                                      /// Send and receive border values in snap (overlapped with interior computation)
    unsigned lookupLocalGhostIndex(ArbLatticeConnectivity::Index gid) const;                                                       /// For a given ghost gid, look up its local id
    void debugDumpConnect(const std::string& name) const;                                                                          /// Dump connectivity info for debug purposes
    void debugDumpVTU() const;                                                                                                     /// Dump VTU info info for debug purposes
};

#endif  // ARBLATTICE_HPP
