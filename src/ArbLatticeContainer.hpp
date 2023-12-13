#ifndef ARBLATTICECONTAINER_HPP
#define ARBLATTICECONTAINER_HPP

#include "Lists.h"
#include "types.h"

/// Container for all data needed on the GPU to perform an iteration
/// Note that all pointers are non-owning, ArbLatticeContainer has strictly view-like semantics
/// Naming of members corresponds to that of ArbLattice
struct ArbLatticeContainer {
    const unsigned* nbrs;
    const real_t* coords;
    const storage_t* snap_in;
    storage_t* snap_out;
#ifdef ADJOINT
    const storage_t* adj_snap_in;
    storage_t* adj_snap_out;
#endif
    const flag_t* node_types;
    unsigned nbrs_pitch, coords_pitch, snaps_pitch, num_border_nodes, num_interior_nodes;

    // Packing/unpacking on device
    storage_t* pack_buf;
    const storage_t* unpack_buf;
    const size_t* pack_inds;
    const size_t* unpack_inds;
    unsigned int pack_sz, unpack_sz;

    // Utilities to facilitate the dynamic lookup of the offset direction index
    int dynamic_offset_lookup_table[Model_m::stencil_box_sz];
    OffsetDir stencil_offset, stencil_size;
};

static_assert(std::is_trivially_copyable<ArbLatticeContainer>::value, "ArbLatticeContainer must be trivially copyable");

#endif  // ARBLATTICECONTAINER_HPP
