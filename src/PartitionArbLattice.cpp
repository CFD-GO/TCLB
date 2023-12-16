#include "PartitionArbLattice.hpp"

#include "../config.h"

#ifdef WITH_PARMETIS

// NOTE: ParMETIS' typedef of `real_t` clashes with TCLB's, ParMETIS facilities need to live in a very weakly coupled translation unit
#include <parmetis.h>

#include <algorithm>
#include <cassert>
#include <functional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "UtilTypes.hpp"
#include "mpitools.hpp"

namespace detail {
auto convertDistToParmetisInts(const std::vector<long>& dist) -> std::vector<idx_t> {
    return std::vector<idx_t>(dist.cbegin(), dist.cend());
}

auto convertDistFromParmetisInts(const std::vector<idx_t>& dist) -> std::vector<long> {
    return std::vector<long>(dist.cbegin(), dist.cend());
}

bool dirWeightsAreEqual(const std::vector<size_t>& dir_wgts) {
    return std::adjacent_find(dir_wgts.begin(), dir_wgts.end(), std::not_equal_to{}) == dir_wgts.end();
}

struct ParmetisGraph {
    CrsGraph<idx_t, idx_t> graph;
    std::vector<idx_t> vert_wgts, edge_wgts, vert_dist;
};

static auto toParmetisFormat(const ArbLatticeConnectivity& connectivity, const std::vector<size_t>& dir_wgts, size_t comm_size) -> ParmetisGraph {
    auto row_sizes = std::vector<idx_t>(connectivity.getLocalSize(), connectivity.Q);
    for (size_t node = connectivity.chunk_begin; node != connectivity.chunk_end; ++node) {
        size_t i = node - connectivity.chunk_begin;
        for (size_t q = 0; q != connectivity.Q; ++q) {
            auto nbr = connectivity.neighbor(q, i);
            if (nbr == -1 || static_cast<size_t>(nbr) == node) --row_sizes[i];
        }
            
    }
    const auto row_sz_span = Span(row_sizes.cbegin(), row_sizes.cend());
    auto parmetis_graph = CrsGraph(row_sz_span);
    std::vector<idx_t> node_wgts, edge_wgts;
    // TODO: Node weights
    const bool edge_wgts_eq = dirWeightsAreEqual(dir_wgts);
    if (!edge_wgts_eq) edge_wgts.reserve(parmetis_graph.numEntries());
    std::vector<size_t> ind_tab(connectivity.Q);
    std::vector<idx_t> temp(connectivity.Q);
    for (size_t  node = connectivity.chunk_begin; node != connectivity.chunk_end; ++node) {
        size_t local_node_ind = node - connectivity.chunk_begin;
        auto graph_row = parmetis_graph.getRow(static_cast<idx_t>(local_node_ind));
        size_t i = 0;
        for (size_t nbr_ind = 0; nbr_ind != connectivity.Q; ++nbr_ind) {
            const auto nbr = connectivity.neighbor(nbr_ind, local_node_ind);
            if (nbr != -1 && static_cast<size_t>(nbr) != node) {
                graph_row[i] = static_cast<idx_t>(nbr);
                i++;
                if (!edge_wgts_eq) edge_wgts.push_back(static_cast<idx_t>(dir_wgts[nbr_ind]));
            }
        }
        assert(i == graph_row.size());
        // if (edge_wgts_eq) std::sort(graph_row.begin(), graph_row.end());
        // else {  // We need to permute the edge targets (i.e. nodes on the other side of the edge) and edge weights together
        //     const auto node_inds = Span(ind_tab.data(), graph_row.size());
        //     std::iota(node_inds.begin(), node_inds.end(), 0);
        //     std::sort(node_inds.begin(), node_inds.end(), [&](size_t i1, size_t i2) { return graph_row[i1] < graph_row[i2]; });
        //     std::transform(node_inds.begin(), node_inds.end(), temp.begin(), [&](size_t i) { return graph_row[i]; });
        //     std::copy_n(temp.begin(), graph_row.size(), graph_row.begin());
        //     auto this_nodes_edge_wgts_begin = std::prev(edge_wgts.end(), graph_row.size());
        //     std::transform(node_inds.begin(), node_inds.end(), temp.begin(), [&](size_t i) { return this_nodes_edge_wgts_begin[i]; });
        //     std::copy_n(temp.begin(), graph_row.size(), this_nodes_edge_wgts_begin);
        // }
        ++local_node_ind;
    }
    return {std::move(parmetis_graph), std::move(node_wgts), std::move(edge_wgts), convertDistToParmetisInts(computeInitialNodeDist(connectivity.num_nodes_global, comm_size))};
}

auto makeTransposedCoordsForParmetis(const ArbLatticeConnectivity& connect) -> std::unique_ptr<real_t[]> {
    const auto n_nodes = connect.getLocalSize();
    auto retval = std::make_unique<real_t[]>(n_nodes * 3);
    for (size_t dim = 0; dim != 3; ++dim)
        for (size_t n = 0; n != n_nodes; ++n) retval[3 * n + dim] = static_cast<real_t>(connect.coord(dim, n));
    return retval;
}

struct PartitioningResult {
    std::vector<idx_t> part;
    idx_t edgecut;
};

inline auto invokeParmetisPartitioner(const ParmetisGraph& dist_graph, const std::unique_ptr<real_t[]>& coords, MPI_Comm comm, bool edges_are_weighted, bool verts_are_weighted) -> PartitioningResult {
    int comm_rank{}, comm_size{};
    MPI_Comm_rank(comm, &comm_rank);
    MPI_Comm_size(comm, &comm_size);

    const auto& [graph, vert_wgts, edge_wgts, node_dist] = dist_graph;
    idx_t wgt_flag = 2 * verts_are_weighted + edges_are_weighted;
    idx_t numflag = 0;
    idx_t ndims = 3;
    idx_t ncon = 1;
    idx_t nparts = comm_size;
    auto tpwgts = std::vector<real_t>(nparts);
    std::fill(tpwgts.begin(), tpwgts.end(), 1. / static_cast<real_t>(nparts));
    real_t ubvec = 1.05f;
    idx_t options = 0;

    PartitioningResult retval;
    retval.part.resize(graph.numRows());

    if (ParMETIS_V3_PartGeomKway(const_cast<idx_t*>(node_dist.data()), graph.getRawOffsets().data(), graph.getRawEntries().data(), verts_are_weighted ? const_cast<idx_t*>(vert_wgts.data()) : nullptr, edges_are_weighted ? const_cast<idx_t*>(edge_wgts.data()) : nullptr, &wgt_flag, &numflag, &ndims,
                                 coords.get(), &ncon, &nparts, tpwgts.data(), &ubvec, &options, &retval.edgecut, retval.part.data(), &comm) != METIS_OK)
        throw std::runtime_error("Error in \"ParMETIS_V3_PartGeomKway\"");
    return retval;
}

struct RefinementStageData {
    ParmetisGraph graph;         // (weighted) graph in dist-CRS format
    std::vector<size_t> og_ids;  // GIDs of the vertices from the lattice which was initially read (we need to keep track of them to correctly distribute direction, zone, and coordinate info at the end)
};

inline void mpiWaitAll(std::vector<MPI_Request>& reqs) {
    MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
}

inline void redistributeParmetisGraph(RefinementStageData& data, const std::vector<idx_t>& part, MPI_Comm comm, bool edges_are_weighted, bool verts_are_weighted) {
    int comm_rank{}, comm_size{};
    MPI_Comm_rank(comm, &comm_rank);
    MPI_Comm_size(comm, &comm_size);

    auto& [parmetis_graph, og_ids] = data;
    auto& [graph, vert_wgts, edge_wgts, vert_dist] = parmetis_graph;

    // MPI args
    constexpr auto tags = std::array{0, 1, 2, 3, 4, 5};
    const auto [vi_tag, e_tag, es_tag, vw_tag, ew_tag, gid_tag] = tags;
    const auto metis_mpi_int_t = mpitools::getMPIType<idx_t>();

    static_assert(std::is_same_v<int, idx_t>);
    static_assert(std::is_same_v<unsigned long, size_t>);

    // Count how many verts and edges we have to send and receive
    std::vector<unsigned> src_sz(comm_size * 2), dest_sz(comm_size * 2);  // Number of vertices + sum of their degrees
    for (size_t i = 0; i<part.size(); i++) {
        auto dest = part[i];
        dest_sz[dest * 2] += 1;
        dest_sz[dest * 2 + 1] += static_cast<unsigned>(graph.getRow(i).size());
    }
    MPI_Alltoall(dest_sz.data(), 2, MPI_UNSIGNED, src_sz.data(), 2, MPI_UNSIGNED, comm);  // This has to be blocking, everything else depends on the result
    constexpr auto count_nnz_verts = [](const std::vector<unsigned>& v) {
        size_t retval = 0;
        for (size_t i = 0; i < v.size(); i += 2) retval += (v[i] > 0);  // count only even inds, i.e. vertex counts
        return retval;
    };

    // Requests vector: 4 per send + 4 per recv + 1 collective
    std::vector<MPI_Request> reqs;
    auto get_req = [&reqs, req_ind = 0]() mutable { reqs.push_back(MPI_REQUEST_NULL); return &reqs.back(); };

    // Store the local base vertex index so that we can start gathering the new distribution into the existing vector
    const idx_t my_base_old = vert_dist[comm_rank];

    // Start computing new vertex distribution
    idx_t n_verts_new = 0, n_edges_new = 0;
    for (int i = 0; i != comm_size; ++i) {
        n_verts_new += src_sz[2 * i];
        n_edges_new += src_sz[2 * i + 1];
    }
    MPI_Iallgather(&n_verts_new, 1, metis_mpi_int_t, std::next(vert_dist.data()), 1, metis_mpi_int_t, comm, get_req());

    // Data structure for storing comm buffers (the graph from ParmetisGraph is stored explicitly as 2 vectors)
    struct NbrInfo {
        std::vector<idx_t> vert_ids, edges, edge_sizes, vert_wgts, edge_wgts;
        std::vector<size_t> vert_gids;
    };
    std::unordered_map<int, NbrInfo> in_map, out_map;

    // Allocate buffers for comms, post receives
    for (int nbr_rank = 0; nbr_rank != comm_size; ++nbr_rank) {
        const auto in_sz = src_sz[nbr_rank * 2], in_edges_sz = src_sz[2 * nbr_rank + 1], out_sz = dest_sz[nbr_rank * 2], out_edges_sz = dest_sz[2 * nbr_rank + 1];
        if (in_sz != 0) {
            auto& [vi, e, es, vw, ew, gids] = in_map[nbr_rank];
            mpitools::MPI_Irecv(vi, in_sz, nbr_rank, vi_tag, comm, get_req());
            mpitools::MPI_Irecv(e, in_edges_sz, nbr_rank, e_tag, comm, get_req());
            mpitools::MPI_Irecv(es, in_sz, nbr_rank, es_tag, comm, get_req());
            mpitools::MPI_Irecv(vw, in_sz * verts_are_weighted, nbr_rank, vw_tag, comm, get_req());
            mpitools::MPI_Irecv(ew, in_edges_sz * edges_are_weighted, nbr_rank, ew_tag, comm, get_req());
            mpitools::MPI_Irecv(gids, in_sz, nbr_rank, gid_tag, comm, get_req());
        }
        if (out_sz != 0) {
            auto& [vi, e, es, vw, ew, gids] = out_map[nbr_rank];
            vi.reserve(out_sz);
            e.reserve(out_edges_sz);
            es.reserve(out_sz);
            vw.reserve(in_sz * verts_are_weighted);
            ew.reserve(in_edges_sz * edges_are_weighted);
            gids.reserve(out_sz);
        }
    }

    // Pack send data
    for (idx_t local_vert_ind = 0, edge_ind = 0; local_vert_ind != data.graph.graph.numRows(); ++local_vert_ind) {
        auto& [vi, e, es, vw, ew, gids] = out_map.at(part[local_vert_ind]);
        vi.push_back(local_vert_ind + my_base_old);
        const auto edges = graph.getRow(local_vert_ind);
        std::copy(edges.begin(), edges.end(), std::back_inserter(e));
        es.push_back(static_cast<idx_t>(edges.size()));
        if (verts_are_weighted) vw.push_back(vert_wgts[local_vert_ind]);
        if (edges_are_weighted) {
            std::copy_n(std::next(edge_wgts.cbegin(), edge_ind), edges.size(), std::back_inserter(ew));
            edge_ind += static_cast<idx_t>(edges.size());
        }
        gids.push_back(og_ids[local_vert_ind]);
    }

    // Post sends
    for (const auto& [dest_rank, info] : out_map) {
        const auto& [vi, e, es, vw, ew, gids] = info;
        mpitools::MPI_Isend(vi, dest_rank, vi_tag, comm, get_req());
        mpitools::MPI_Isend(e, dest_rank, e_tag, comm, get_req());
        mpitools::MPI_Isend(es, dest_rank, es_tag, comm, get_req());
        mpitools::MPI_Isend(vw, dest_rank, vw_tag, comm, get_req());
        mpitools::MPI_Isend(ew, dest_rank, ew_tag, comm, get_req());
        mpitools::MPI_Isend(gids, dest_rank, gid_tag, comm, get_req());
    }

    // Wait for comms to complete
    mpiWaitAll(reqs);

    // Finish computing new vertex distribution
    std::inclusive_scan(vert_dist.begin(), vert_dist.end(), vert_dist.begin());

    // Consolidate vertex data
    vert_wgts.clear();
    edge_wgts.clear();
    og_ids.clear();
    vert_wgts.reserve(n_verts_new * verts_are_weighted);
    edge_wgts.reserve(n_edges_new * edges_are_weighted);
    og_ids.reserve(n_verts_new);
    std::vector<idx_t> vert_degrees_new, vert_old_ids;
    vert_degrees_new.reserve(n_verts_new);
    vert_old_ids.reserve(n_verts_new);
    for (const auto& [rank, info] : in_map) {
        const auto& [vi, e, es, vw, ew, gids] = info;
        std::copy(vi.cbegin(), vi.cend(), std::back_inserter(vert_old_ids));
        std::copy(es.cbegin(), es.cend(), std::back_inserter(vert_degrees_new));
        std::copy(vw.cbegin(), vw.cend(), std::back_inserter(vert_wgts));
        std::copy(ew.cbegin(), ew.cend(), std::back_inserter(edge_wgts));
        std::copy(gids.cbegin(), gids.cend(), std::back_inserter(og_ids));
    }
    graph = CrsGraph(Span(vert_degrees_new.cbegin(), vert_degrees_new.cend()));
    idx_t* out_ptr = graph.getRawEntries().data();
    for (const auto& [rank, info] : in_map) out_ptr = std::copy(info.edges.cbegin(), info.edges.cend(), out_ptr);

    // Reindex edge targets based on the numbering scheme after the redistribution
    MPI_Barrier(comm);
    auto old_to_new_index_map = std::unordered_map<idx_t, idx_t>{};
    for (idx_t i = 0; i != n_verts_new; ++i) old_to_new_index_map.emplace(vert_old_ids[i], i + vert_dist[comm_rank]);
    auto ghost_verts = std::unordered_set<idx_t>{};
    for (idx_t v : graph.getRawEntries())
        if (old_to_new_index_map.find(v) == old_to_new_index_map.end()) ghost_verts.insert(v);
    std::vector<idx_t> comm_buf, to_delete;
    for (int bcast_rank = 0; bcast_rank != comm_size; ++bcast_rank) {
        const idx_t base = vert_dist[bcast_rank];
        const idx_t sz = vert_dist[bcast_rank + 1] - base;
        if (comm_rank == bcast_rank) {
            MPI_Bcast(vert_old_ids.data(), sz, metis_mpi_int_t, bcast_rank, comm);
        } else {
            comm_buf.resize(sz);
            MPI_Bcast(comm_buf.data(), sz, metis_mpi_int_t, bcast_rank, comm);  // TODO: overlap communication with processing the previous buffer
            to_delete.clear();
            if (ghost_verts.empty()) continue;
            idx_t new_id = base;
            for (auto old_id : comm_buf) {
                const auto it = ghost_verts.find(old_id);
                if (it != ghost_verts.end()) {
                    old_to_new_index_map.emplace(old_id, new_id);
                    to_delete.push_back(old_id);
                }
                ++new_id;
            }
            for (idx_t d : to_delete) ghost_verts.erase(d);
        }
    }

    MPI_Barrier(comm);
    for (idx_t& v : graph.getRawEntries()) v = old_to_new_index_map.at(v);
}

inline auto invokeParmetisRepartitioner(const ParmetisGraph& dist_graph, MPI_Comm comm, bool edges_are_weighted, bool verts_are_weighted) -> PartitioningResult {
    int comm_rank{}, comm_size{};
    MPI_Comm_rank(comm, &comm_rank);
    MPI_Comm_size(comm, &comm_size);

    const auto& [graph, node_wgts, edge_wgts, node_dist] = dist_graph;
    idx_t wgt_flag = 2 * (!node_wgts.empty()) + !edge_wgts.empty();
    idx_t numflag = 0;
    idx_t ncon = 1;
    idx_t nparts = comm_size;
    auto tpwgts = std::vector<real_t>(nparts);
    std::fill(tpwgts.begin(), tpwgts.end(), 1. / static_cast<real_t>(nparts));
    real_t ubvec = 1.05f;
    idx_t options = 0;

    PartitioningResult retval;
    retval.part.resize(graph.numRows());

    if (ParMETIS_V3_RefineKway(const_cast<idx_t*>(node_dist.data()), graph.getRawOffsets().data(), graph.getRawEntries().data(), verts_are_weighted ? const_cast<idx_t*>(node_wgts.data()) : nullptr, edges_are_weighted ? const_cast<idx_t*>(edge_wgts.data()) : nullptr, &wgt_flag, &numflag, &ncon,
                               &nparts, tpwgts.data(), &ubvec, &options, &retval.edgecut, retval.part.data(), &comm) != METIS_OK)
        throw std::runtime_error("Error in \"ParMETIS_V3_RefineKway\"");
    return retval;
}

auto recoverConnectivity(const ArbLatticeConnectivity& connectivity_initial, const RefinementStageData& refine_data_final, MPI_Comm comm, size_t self_edge_ind = -1) -> std::pair<ArbLatticeConnectivity, std::vector<long>> {
    int comm_rank{}, comm_size{};
    MPI_Comm_rank(comm, &comm_rank);
    MPI_Comm_size(comm, &comm_size);

    const auto& [parmetis_graph, og_ids] = refine_data_final;
    const auto& [graph, vert_wgts, edge_wgts, vert_dist] = parmetis_graph;

    const auto og_dist = computeInitialNodeDist(connectivity_initial.num_nodes_global, comm_size);
    const auto compute_original_rank = [&](auto gid) {
        const auto slot_end_iter = std::upper_bound(og_dist.cbegin(), og_dist.cend(), gid);
        return static_cast<int>(std::distance(og_dist.cbegin(), slot_end_iter) - 1);
    };

    // Tell the original owners how many of their nodes I need info on
    // Note on naming: the prefixes in- and out- refer to the direction the node data will be sent. We first need to query the required IDs, which means
    // we'll be sending to in-nbrs and receiving from out-nbrs. This may be a bit confusing, but it is correct.
    std::vector<unsigned> in_sz(comm_size), out_sz(comm_size);
    std::unordered_map<int, std::vector<unsigned long>> in_gid_map, out_gid_map;
    for (auto gid : og_ids) {
        const auto og_owner = compute_original_rank(gid);
        ++in_sz.at(og_owner);
        in_gid_map[og_owner].push_back(gid);
    }
    MPI_Alltoall(in_sz.data(), 1, MPI_UNSIGNED, out_sz.data(), 1, MPI_UNSIGNED, comm);

    const auto num_out_nbrs = std::count_if(out_sz.cbegin(), out_sz.cend(), [](auto s) { return s != 0; });
    std::vector<MPI_Request> reqs(in_gid_map.size() + num_out_nbrs, MPI_REQUEST_NULL);
    int ri = 0;
    const auto get_req = [&reqs, &ri] { return &reqs.at(ri++); };

    // Communicate GIDs of required nodes
    {
        int src_rank = 0;
        for (auto out : out_sz) {
            if (out != 0) {
                auto& out_vec = out_gid_map[src_rank];
                out_vec.resize(out);
                MPI_Irecv(out_vec.data(), out, MPI_UNSIGNED_LONG, src_rank, 0, comm, get_req());
            }
            ++src_rank;
        }
    }
    for (const auto& [dest_rank, dest_vec] : in_gid_map) {
        assert(dest_vec.size() == in_sz[dest_rank]);
        MPI_Isend(dest_vec.data(), in_sz[dest_rank], MPI_UNSIGNED_LONG, dest_rank, 0, comm, get_req());
    }
    mpiWaitAll(reqs);

    constexpr auto tags = std::array{0, 1, 2, 3};
    const auto [coord_tag, nbr_tag, zone_sz_tag, zone_tag] = tags;
    reqs.resize((in_gid_map.size() + num_out_nbrs) * tags.size(), MPI_REQUEST_NULL);

    // Pack and communicate node data
    struct ArbLatticeNodeData {
        std::vector<double> coords;
        std::vector<char> nbr_bmps;
        std::vector<typename ArbLatticeConnectivity::ZoneIndex> zone_sz, zones;
    };
    std::vector<size_t> zone_sz_offsets(connectivity_initial.getLocalSize());
    std::transform_exclusive_scan(connectivity_initial.zones_per_node.get(), std::next(connectivity_initial.zones_per_node.get(), connectivity_initial.getLocalSize()), zone_sz_offsets.begin(), size_t{0}, std::plus{}, [](auto v) -> size_t { return v; });
    const auto pack_data = [&](const std::vector<unsigned long>& gids) {
        ArbLatticeNodeData retval{};
        auto& [coords, nbr_bmps, zone_sz, zones] = retval;
        coords.reserve(gids.size() * 3);
        nbr_bmps.reserve(gids.size() * connectivity_initial.Q);
        zone_sz.reserve(gids.size());
        zones.reserve(gids.size());
        for (auto gid : gids) {
            const auto lid = gid - og_dist[comm_rank];
            for (size_t dim = 0; dim != 3; ++dim) coords.push_back(connectivity_initial.coord(dim, lid));
            for (size_t qi = 0; qi != connectivity_initial.Q; ++qi) nbr_bmps.push_back(connectivity_initial.neighbor(qi, lid) != -1);
            zone_sz.push_back(connectivity_initial.zones_per_node[lid]);
            std::copy_n(std::next(connectivity_initial.zones.begin(), zone_sz_offsets[lid]), zone_sz.back(), std::back_inserter(zones));
        }
        return retval;
    };
    std::unordered_map<int, ArbLatticeNodeData> in_data_map, out_data_map;
    ri = 0;
    for (const auto& [src_rank, in_ids] : in_gid_map) {
        auto& [coords, nbr_bmps, zone_sz, zones] = in_data_map[src_rank];
        mpitools::MPI_Irecv(coords, in_ids.size() * 3, src_rank, coord_tag, comm, get_req());
        mpitools::MPI_Irecv(nbr_bmps, in_ids.size() * connectivity_initial.Q, src_rank, nbr_tag, comm, get_req());
        mpitools::MPI_Irecv(zone_sz, in_ids.size(), src_rank, zone_sz_tag, comm, get_req());
    }
    for (const auto& [dest_rank, out_ids] : out_gid_map) {
        auto& out = out_data_map[dest_rank];
        out = pack_data(out_ids);
        auto& [coords, nbr_bmps, zone_sz, zones] = out;
        mpitools::MPI_Isend(coords, dest_rank, coord_tag, comm, get_req());
        mpitools::MPI_Isend(nbr_bmps, dest_rank, nbr_tag, comm, get_req());
        mpitools::MPI_Isend(zone_sz, dest_rank, zone_sz_tag, comm, get_req());
        mpitools::MPI_Isend(zones, dest_rank, zone_tag, comm, get_req());
    }
    std::vector<bool> probed(in_gid_map.size());
    size_t num_probed = 0;
    while (num_probed != in_gid_map.size()) {
        size_t i = 0;
        for (const auto& [src_rank, in_ids] : in_gid_map) {
            if (!probed[i]) {
                MPI_Status status;
                int ready_flag;
                MPI_Iprobe(src_rank, zone_tag, comm, &ready_flag, &status);
                if (ready_flag) {
                    int count;
                    MPI_Get_elements(&status, mpitools::getMPIType<typename ArbLatticeConnectivity::ZoneIndex>(), &count);
                    mpitools::MPI_Irecv(in_data_map.at(src_rank).zones, count, src_rank, zone_tag, comm, get_req());
                    ++num_probed;
                    probed[i] = true;
                }
            }
            ++i;
        }
    }
    mpiWaitAll(reqs);

    auto connectivity_new = ArbLatticeConnectivity(vert_dist[comm_rank], vert_dist[comm_rank + 1], connectivity_initial.num_nodes_global, connectivity_initial.Q);
    connectivity_new.grid_size = connectivity_initial.grid_size;
    std::vector<size_t> rank_inds(comm_size), rank_zone_inds(comm_size);
    const auto unpack_data = [&](idx_t lid, int og_gid) {
        const auto og_owner = compute_original_rank(og_gid);
        const auto& [coords, nbr_bmp, zone_sz, zones] = in_data_map.at(og_owner);
        const size_t i_node = rank_inds[og_owner]++;
        for (size_t dim = 0; dim != 3; ++dim) connectivity_new.coord(dim, lid) = coords[dim + i_node * 3];
        connectivity_new.og_index[lid] = og_gid;
        const auto edges = graph.getRow(lid);
        const auto gid = vert_dist[comm_rank] + lid;
        size_t active_ind = 0;
        for (size_t qi = 0; qi != connectivity_initial.Q; ++qi) {
            if (qi == self_edge_ind) {
                connectivity_new.neighbor(qi, lid) = gid;
            } else if (!nbr_bmp[qi + i_node * connectivity_initial.Q]) {
                connectivity_new.neighbor(qi, lid) = -1;
            } else {
                connectivity_new.neighbor(qi, lid) = edges[active_ind];
                active_ind++;
            }
        }
        assert(active_ind == edges.size());
        auto& zs = connectivity_new.zones_per_node[lid];
        zs = zone_sz[i_node];
        size_t& i_zone = rank_zone_inds[og_owner];
        std::copy_n(std::next(zones.begin(), i_zone), zs, std::back_inserter(connectivity_new.zones));
        i_zone += zs;
    };
    idx_t lid = 0;
    for (auto og_gid : og_ids) {
        unpack_data(lid++, og_gid);
    }
    return std::make_pair(std::move(connectivity_new), convertDistFromParmetisInts(vert_dist));
}

inline auto getDefaultStopCriterion() {
    return [prev = -1](idx_t edgecut) mutable {
        constexpr double improvement_threshold = .001;  // Iterate until relative improvement in comm volume drops below this value
        const auto impr = static_cast<double>(prev - edgecut) / static_cast<double>(prev);
        prev = edgecut;
        return impr < improvement_threshold;
    };
}

template <typename StopCrit = decltype(getDefaultStopCriterion())>
auto partitionLattice(const ArbLatticeConnectivity& connectivity, const std::vector<size_t>& dir_wgts, MPI_Comm comm, size_t self_edge_ind, std::vector<PartOutput::LoggedEvent>& log, StopCrit&& stop_criterion = getDefaultStopCriterion()) -> std::pair<ArbLatticeConnectivity, std::vector<long>> {
    int comm_rank{}, comm_size{};
    MPI_Comm_rank(comm, &comm_rank);
    MPI_Comm_size(comm, &comm_size);

    const bool nodes_are_weighted = false;
    const bool edges_are_weighted = !dirWeightsAreEqual(dir_wgts);

    RefinementStageData refine_data;
    auto& [graph, og_inds] = refine_data;

    const auto coords_aos = makeTransposedCoordsForParmetis(connectivity);
    graph = toParmetisFormat(connectivity, dir_wgts, comm_size);
    auto [part_initial, edgecut_initial] = invokeParmetisPartitioner(graph, coords_aos, comm, edges_are_weighted, nodes_are_weighted);
    log.push_back(PartOutput::LoggedEvent{PartOutput::MsgType::Notice, "Initial partitioning has edgecut value: " + std::to_string(edgecut_initial)});

    og_inds.resize(graph.graph.numRows());
    std::iota(og_inds.begin(), og_inds.end(), graph.vert_dist[comm_rank]);
    redistributeParmetisGraph(refine_data, part_initial, comm, edges_are_weighted, nodes_are_weighted);

    idx_t edgecut = edgecut_initial;
    while (!stop_criterion(edgecut)) {
        const auto [part, edgecut_current] = invokeParmetisRepartitioner(graph, comm, edges_are_weighted, nodes_are_weighted);
        redistributeParmetisGraph(refine_data, part, comm, edges_are_weighted, nodes_are_weighted);
        log.push_back(PartOutput::LoggedEvent{PartOutput::MsgType::Notice, "Edgecut after refinement: " + std::to_string(edgecut_current)});
        edgecut = edgecut_current;
    }

    return recoverConnectivity(connectivity, refine_data, comm, self_edge_ind);
}
}  // namespace detail

PartOutput partitionArbLattice(ArbLatticeConnectivity& lattice, const std::vector<size_t>& dir_wgts, size_t self_edge_ind, MPI_Comm comm) {
    PartOutput retval;
    try {
        auto [connect_new, dist_new] = detail::partitionLattice(lattice, dir_wgts, comm, self_edge_ind, retval.event_log);
        retval.partition_distribution = std::move(dist_new);
        lattice = std::move(connect_new);
        return retval;
    } catch (const std::exception& e) { retval.event_log.push_back(PartOutput::LoggedEvent{PartOutput::MsgType::Error, e.what()}); }
    return retval;
}

#else

PartOutput partitionArbLattice(ArbLatticeConnectivity& lattice, const std::vector<size_t>& dir_wgts, size_t self_edge_ind, MPI_Comm comm) {
    int comm_size;
    MPI_Comm_size(comm, &comm_size);
    PartOutput retval;
    retval.partition_distribution = computeInitialNodeDist(lattice.num_nodes_global, comm_size);
    retval.event_log.push_back(PartOutput::LoggedEvent{PartOutput::MsgType::Warning, "ParMETIS was not enabled, the quality of the partition may be very poor"});
    return retval;
}

#endif