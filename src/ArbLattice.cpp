#include "ArbLattice.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <fstream>
#include <functional>
#include <numeric>
#include <optional>
#include <unordered_set>

#include "GetThreads.h"
#include "PartitionArbLattice.hpp"
#include "mpitools.hpp"
#include "pinned_allocator.hpp"
#include "vtuOutput.h"

ArbLattice::ArbLattice(size_t num_snaps_, const UnitEnv& units_, const std::map<std::string, int>& setting_zones, pugi::xml_node arb_node, MPI_Comm comm_)
    : LatticeBase(ZONESETTINGS, ZONE_MAX, num_snaps_, units_), comm(comm_) {
    initialize(num_snaps_, setting_zones, arb_node);
}

void ArbLattice::initialize(size_t num_snaps_, const std::map<std::string, int>& setting_zones, pugi::xml_node arb_node) {
    const int rank = mpitools::MPI_Rank(comm);
    sizes.snaps = num_snaps_;
#ifdef ADJOINT
    sizes.snaps += 2;  // Adjoint snaps are appended to the total snap allocation
#endif
    initialized_from = arb_node;
    const auto debug_attr = arb_node.attribute("debug");
    if (debug_attr) debug_name = debug_attr.value();

    const auto name_attr = arb_node.attribute("file");
    if (!name_attr) throw std::runtime_error{"The ArbitraryLattice node lacks the \"file\" attribute"};
    const std::string cxn_path = name_attr.value();
    readFromCxn(cxn_path);
    global_node_dist = computeInitialNodeDist(connect.num_nodes_global, mpitools::MPI_Size(comm));
    debugDumpConnect("conn_before");
    partition();
    debugDumpConnect("conn_after");
    if (connect.getLocalSize() == 0)
        throw std::runtime_error{"At least one MPI rank has an empty partition, please use fewer MPI ranks"};  // Realistically, this should never happen
    computeGhostNodes();
    computeLocalPermutation(arb_node, setting_zones);
    allocDeviceMemory();
    initDeviceData(arb_node, setting_zones);
    local_bounding_box = getLocalBoundingBox();
    vtu_geom = makeVTUGeom();
    debugDumpVTU();
    initCommManager();
    initContainer();

    debug1("Initialized arbitrary lattice with: border nodes=%lu; interior nodes=%lu; ghost nodes=%lu",
           sizes.border_nodes,
           getLocalSize() - sizes.border_nodes,
           ghost_nodes.size());
}

int ArbLattice::reinitialize(size_t num_snaps_, const std::map<std::string, int>& setting_zones, pugi::xml_node arb_node) {
    size_t adjoint_snaps = 0;
#ifdef ADJOINT
    adjoint_snaps += 2;
#endif
    if (num_snaps_ + adjoint_snaps != sizes.snaps || arb_node != initialized_from) try {
            initialize(num_snaps_, setting_zones, arb_node);
        } catch (const std::exception& e) {
            ERROR(e.what());
            return EXIT_FAILURE;
        }
    return EXIT_SUCCESS;
}

void ArbLattice::readFromCxn(const std::string& cxn_path) {
    using namespace std::string_literals;
    using namespace std::string_view_literals;

    const int comm_rank = mpitools::MPI_Rank(comm), comm_size = mpitools::MPI_Size(comm);

    // Open file + utils for error reporting
    std::fstream file(cxn_path, std::ios_base::in);
    const auto wrap_err_msg = [&](std::string msg) {
        return "Error while reading "s.append(cxn_path.c_str()).append(" on MPI rank ").append(std::to_string(comm_rank)).append(": ").append(msg);
    };
    const auto check_file_ok = [&](const std::string& err_message = "Unknown error") {
        if (!file) throw std::ios_base::failure(wrap_err_msg(err_message));
    };
    const auto check_expected_word = [&](std::string_view expected, std::string_view actual) {
        if (expected != actual) {
            const auto err_msg = "Unexpected section header: "s.append(actual).append("; Expected: ").append(expected);
            throw std::logic_error(wrap_err_msg(err_msg));
        }
    };
    check_file_ok("Could not open file");

    // Implement the pattern: 1) read header 2) read size 3) invoke body on the read size, return result
    std::string word;
    const auto process_section = [&](std::string_view header, auto&& body) {
        file >> word;
        check_file_ok("Failed to read section header: "s.append(header));
        check_expected_word(header, word);
        size_t size{};
        file >> size;
        check_file_ok("Failed to read section size: "s.append(header));
        if constexpr (std::is_same_v<void, std::invoke_result_t<decltype(body), size_t>>) {
            body(size);
            check_file_ok("Failed to read section: "s.append(header));
        } else {
            auto retval = body(size);
            check_file_ok("Failed to read section: "s.append(header));
            return retval;
        }
    };

    // Offset directions
    const auto q_provided = process_section("OFFSET_DIRECTIONS", [&file](size_t n_q_provided) {
        std::vector<OffsetDir> retval;
        retval.reserve(n_q_provided);
        for (size_t i = 0; i != n_q_provided; ++i) {
            OffsetDir dirs{};
            file >> dirs.x >> dirs.y >> dirs.z;
            retval.push_back(dirs);
        }
        return retval;
    });

    // Check all required dirs are present, construct the lookup table
    std::array<size_t, Q> req_prov_perm{};  // Lookup table between provided and required dirs
    {
        size_t i = 0;
        for (const auto& req : Model_m::offset_directions) {
            const auto prov_it = std::find(q_provided.cbegin(), q_provided.cend(), req);
            if (prov_it == q_provided.cend()) {
                const auto [x, y, z] = req;
                const auto err_msg = "The arbitrary lattice file does not provide the required direction: ["s.append(std::to_string(x))
                                         .append(", ")
                                         .append(std::to_string(y))
                                         .append(", ")
                                         .append(std::to_string(z))
                                         .append("]");
                throw std::runtime_error(wrap_err_msg(err_msg));
            }
            req_prov_perm[i++] = std::distance(q_provided.cbegin(), prov_it);
        }
    }

    // Grid size
    file >> word;
    check_file_ok("Failed to read section header: GRID_SIZE");
    check_expected_word("GRID_SIZE", word);
    double grid_size{};
    file >> grid_size;
    check_file_ok("Failed to read section: GRID_SIZE");

    // Labels present in the .cxn file
    const auto labels = process_section("NODE_LABELS", [&file](size_t n_labels) {
        std::vector<std::string> retval(n_labels);
        for (auto& g : retval) file >> g;
        return retval;
    });
    for (size_t i = 0; i != labels.size(); ++i) label_to_ind_map.emplace(labels[i], i);

    // Nodes header
    process_section("NODES", [&](size_t num_nodes_global) {
        // Compute the current rank's offset and number of nodes to read
        const auto chunk_offsets = computeInitialNodeDist(num_nodes_global, static_cast<size_t>(comm_size));
        const auto chunk_begin = static_cast<size_t>(chunk_offsets[comm_rank]), chunk_end = static_cast<size_t>(chunk_offsets[comm_rank + 1]);
        const auto num_nodes_local = chunk_end - chunk_begin;

        connect = ArbLatticeConnectivity(chunk_begin, chunk_end, num_nodes_global, Q);
        connect.grid_size = grid_size;

        // Skip chunk_begin + 1 (header) newlines
        for (size_t i = 0; i != chunk_begin + 1; ++i) file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        check_file_ok("Failed to skip ahead to this rank's chunk");

        // Parse node data
        std::vector<long> nbrs_in_file(q_provided.size());
        for (size_t local_node_ind = 0; local_node_ind != num_nodes_local; ++local_node_ind) {
            // Read coords
            file >> connect.coord(0, local_node_ind) >> connect.coord(1, local_node_ind) >> connect.coord(2, local_node_ind);

            // Read neighbors, map to local (required) ones
            for (auto& nbr : nbrs_in_file) file >> nbr;
            size_t j = 0;
            for (size_t perm : req_prov_perm) connect.neighbor(j++, local_node_ind) = nbrs_in_file[perm];

            // Read and set zones
            auto& n_zones = connect.zones_per_node[local_node_ind];
            file >> n_zones;
            for (size_t z = 0; z != n_zones; ++z) {
                auto& zone = connect.zones.emplace_back();
                file >> zone;
            }

            check_file_ok("Failed to read node data");
        }
    });
}

void ArbLattice::partition() {
    if (mpitools::MPI_Size(comm) == 1) return;

    const auto zero_dir_ind = std::distance(Model_m::offset_directions.cbegin(),
                                            std::find(Model_m::offset_directions.cbegin(),
                                                      Model_m::offset_directions.cend(),
                                                      OffsetDir{0, 0, 0}));  // Note: the behavior is still correct even if (0,0,0) is not an offset direction
    const auto offset_dir_wgts = std::vector(Model_m::offset_direction_weights.begin(), Model_m::offset_direction_weights.end());
    auto [dist, log] = partitionArbLattice(connect, offset_dir_wgts, zero_dir_ind, comm);
    for (const auto& [type, msg] : log) switch (type) {
            case PartOutput::MsgType::Notice:
                NOTICE(msg.c_str());
                break;
            case PartOutput::MsgType::Warning:
                WARNING(msg.c_str());
                break;
            case PartOutput::MsgType::Error:
                throw std::runtime_error(msg);
        }
    global_node_dist = std::move(dist);
}

void ArbLattice::computeGhostNodes() {
    std::unordered_set<long> ghosts;
    const Span all_nbrs(connect.nbrs.get(), Q * connect.getLocalSize());
    for (auto nbr : all_nbrs)
        if (connect.isGhost(nbr)) ghosts.insert(nbr);
    ghost_nodes.reserve(ghosts.size());
    std::copy(ghosts.cbegin(), ghosts.cend(), std::back_inserter(ghost_nodes));
    std::sort(ghost_nodes.begin(), ghost_nodes.end());
}

std::function<bool(int, int)> ArbLattice::makePermCompare(pugi::xml_node arb_node, const std::map<std::string, int>& setting_zones) {
    // Note: copies of these closures will outlive the function call, careful with the captures (capture by copy)
    const auto get_zyx = [this](int lid) { return std::array{connect.coord(2, lid), connect.coord(1, lid), connect.coord(0, lid)}; };
    const auto get_nt = [this](int lid) { return node_types_host.at(lid); };
    const auto get_nt_zyx = [get_zyx, get_nt](int lid) { return std::make_pair(get_nt(lid), get_zyx(lid)); };
    constexpr auto wrap_projection_as_comparison = [](const auto& proj) { return [proj](int lid1, int lid2) { return proj(lid1) < proj(lid2); }; };

    enum struct PermStrategy { None, Type, Coords, Both };
    static const std::unordered_map<std::string_view, PermStrategy> strat_map = {{"none", PermStrategy::None},
                                                                                 {"type", PermStrategy::Type},
                                                                                 {"coords", PermStrategy::Coords},
                                                                                 {"both", PermStrategy::Both},
                                                                                 {"", PermStrategy::Coords}};  // "" is the default
    const std::string_view strat_str = arb_node.attribute("permutation").value();
    const auto enum_it = strat_map.find(strat_str);
    if (enum_it == strat_map.end())
        throw std::runtime_error{"Unknown permutation strategy for ArbitraryLattice, valid values are: none, type, coords (default), both"};
    const auto strat = enum_it->second;

    if (strat == PermStrategy::Type || strat == PermStrategy::Both) computeNodeTypesOnHost(arb_node, setting_zones, /*permute*/ false);
    switch (strat) {
        case PermStrategy::None:  // Use initial ordering
            return std::less{};
        case PermStrategy::Type:  // Sort by node type only
            return wrap_projection_as_comparison(get_nt);
        case PermStrategy::Coords:  // Sort by coordinates only
            return wrap_projection_as_comparison(get_zyx);
        case PermStrategy::Both:  // Sort by node type, then coordinates
            return wrap_projection_as_comparison(get_nt_zyx);
    }
    return {};  // avoid compiler warning
}

void ArbLattice::computeLocalPermutation(pugi::xml_node arb_node, const std::map<std::string, int>& setting_zones) {
    std::vector<size_t> lids(connect.getLocalSize());  // globalIdx - chunk_begin of elements
    std::iota(lids.begin(), lids.end(), 0);
    const auto is_border_node = [&](int lid) {
        for (size_t q = 0; q != Q; ++q) {
            const auto nbr = connect.neighbor(q, lid);
            if (connect.isGhost(nbr)) return true;
        }
        return false;
    };
    const auto interior_begin = std::stable_partition(lids.begin(), lids.end(), is_border_node);
    sizes.border_nodes = static_cast<size_t>(std::distance(lids.begin(), interior_begin));
    const auto compare = makePermCompare(arb_node, setting_zones);
    std::sort(lids.begin(), interior_begin, compare);
    std::sort(interior_begin, lids.end(), compare);
    local_permutation.resize(connect.getLocalSize());
    size_t i = 0;
    for (const auto& lid : lids) {
        local_permutation[lid] = i;
        i++;
    }
}

void ArbLattice::allocDeviceMemory() {
    // Pitches get updated based on CUDA/HIP padding
    const auto local_sz = connect.getLocalSize();
    sizes.neighbors_pitch = local_sz;
    neighbors_device = cudaMakeUnique2D<unsigned>(sizes.neighbors_pitch, Q);
    sizes.coords_pitch = local_sz;
    coords_device = cudaMakeUnique2D<real_t>(sizes.coords_pitch, 3);
    sizes.snaps_pitch = local_sz + ghost_nodes.size() + 1;
    snaps_device = cudaMakeUnique2D<storage_t>(sizes.snaps_pitch, sizes.snaps * NF);
    node_types_device = cudaMakeUnique<flag_t>(local_sz);
}

std::vector<ArbLattice::NodeTypeBrush> ArbLattice::parseBrushFromXml(pugi::xml_node arb_node, const std::map<std::string, int>& setting_zones) const {
    std::vector<ArbLattice::NodeTypeBrush> retval;
    for (auto node = arb_node.first_child(); node; node = node.next_sibling()) {
        // Requested node type
        const auto ntf_iter =
            std::find_if(model->nodetypeflags.cbegin(), model->nodetypeflags.cend(), [name = node.name()](const auto& ntf) { return ntf.name == name; });
        if (ntf_iter == model->nodetypeflags.cend()) throw std::runtime_error{formatAsString("Unknown node type: %s", node.name())};

        // Determine what kind of node we're parsing and update the brush accordingly
        const auto group_attr = node.attribute("group");
        const auto zone_attr = node.attribute("name");
        if (group_attr) {
            const flag_t mask = ntf_iter->group_flag | (zone_attr ? model->settingzones.flag : 0);
            const flag_t value = ntf_iter->flag | (zone_attr ? (setting_zones.at(zone_attr.value()) << model->settingzones.shift) : 0);
            const std::string group_name = group_attr.value();
            const auto label_iter = label_to_ind_map.find(group_name);
            if (label_iter == label_to_ind_map.end())
                throw std::runtime_error{formatAsString("The required label %s is missing from the .cxn file", group_name)};
            const auto label = label_iter->second;
            const auto has_label = [label](Span<const ArbLatticeConnectivity::ZoneIndex> labels, std::array<double, 3>) {
                return std::find(labels.begin(), labels.end(), label) != labels.end();
            };
            retval.push_back(NodeTypeBrush{has_label, mask, value});
        } else
            throw std::runtime_error{std::string("The ArbitraryLattice XML node contains an incorrectly specified child named ") +
                                     node.name()};  // TODO: implement other node types, e.g. <Box>
    }
    return retval;
}

void ArbLattice::computeNodeTypesOnHost(pugi::xml_node arb_node, const std::map<std::string, int>& setting_zones, bool permute) {
    const auto local_sz = connect.getLocalSize();
    const auto zone_sizes = Span(connect.zones_per_node.get(), local_sz);
    auto zone_offsets = std::vector<size_t>(local_sz);
    std::transform_exclusive_scan(zone_sizes.begin(), zone_sizes.end(), zone_offsets.begin(), size_t{0}, std::plus{}, [](auto label) -> size_t {
        return label;
    });  // labels are stored as a short type, we need to cast it to size_t before computing the scan
    const auto brushes = parseBrushFromXml(arb_node, setting_zones);
    node_types_host = std::pmr::vector<flag_t>(local_sz, &global_pinned_resource);
    for (size_t i = 0; i != local_sz; ++i) {
        const auto labels = Span(std::next(connect.zones.data(), zone_offsets[i]), connect.zones_per_node[i]);
        const auto point = std::array{connect.coord(0, i), connect.coord(1, i), connect.coord(2, i)};
        for (const auto& [pred, mask, val] : brushes)
            if (pred(labels, point)) {
                auto& dest = node_types_host[permute ? local_permutation[i] : i];
                dest = (dest & ~mask) | val;
            }
    }
}

std::pmr::vector<real_t> ArbLattice::computeCoords() const {
    const auto local_sz = connect.getLocalSize();
    std::pmr::vector<real_t> retval(sizes.coords_pitch * 3, &global_pinned_resource);
    for (size_t dim = 0; dim != 3; ++dim) {
        size_t i = 0;
        for (; i != local_sz; ++i) retval[local_permutation[i] + dim * sizes.coords_pitch] = connect.coord(dim, i);
        for (; i != sizes.coords_pitch; ++i) retval[i + dim * sizes.coords_pitch] = std::numeric_limits<real_t>::signaling_NaN();  // padding
    }
    return retval;
}

unsigned int ArbLattice::lookupLocalGhostIndex(ArbLatticeConnectivity::Index gid) const {
    const unsigned local_sz = connect.getLocalSize();
    const auto it = std::lower_bound(ghost_nodes.begin(), ghost_nodes.end(), gid);
    assert(it != ghost_nodes.end());
    assert(*it == gid);
    return local_sz + static_cast<unsigned>(std::distance(ghost_nodes.begin(), it));
}

std::pmr::vector<unsigned> ArbLattice::computeNeighbors() const {
    const auto local_sz = connect.getLocalSize();
    std::pmr::vector<unsigned> retval(sizes.neighbors_pitch * Q, &global_pinned_resource);
    const unsigned invalid_nbr = local_sz + ghost_nodes.size();
    const auto nbr_global_to_local = [&](ArbLatticeConnectivity::Index gid) -> unsigned {
        if (gid == -1) return invalid_nbr;  // dummy row
        else if (connect.isGhost(gid))
            return lookupLocalGhostIndex(gid);
        else
            return local_permutation[gid - connect.chunk_begin];
    };
    for (size_t q = 0; q != Q; ++q) {
        size_t lid = 0;
        for (; lid != local_sz; ++lid) retval[local_permutation[lid] + q * sizes.neighbors_pitch] = nbr_global_to_local(connect.neighbor(q, lid));
        for (; lid != sizes.neighbors_pitch; ++lid) retval[lid + q * sizes.neighbors_pitch] = invalid_nbr;
    }
    return retval;
}

void ArbLattice::initDeviceData(pugi::xml_node arb_node, const std::map<std::string, int>& setting_zones) {
    fillWithStorageNaNAsync(snaps_device.get(), sizes.snaps_pitch * sizes.snaps * NF, inStream);
    computeNodeTypesOnHost(arb_node, setting_zones, /*permute*/ true);
    copyVecToDeviceAsync(node_types_device.get(), node_types_host, inStream);
    const auto nbrs = computeNeighbors();
    copyVecToDeviceAsync(neighbors_device.get(), nbrs, inStream);
    const auto coords = computeCoords();
    copyVecToDeviceAsync(coords_device.get(), coords, inStream);
    CudaStreamSynchronize(inStream);
}

void ArbLattice::initContainer() {
    setSnapIn(0);
    setSnapOut(1);
#ifdef ADJOINT
    setAdjSnapOut(0);
#endif
    launcher.container.nbrs = neighbors_device.get();
    launcher.container.coords = coords_device.get();
    launcher.container.node_types = node_types_device.get();
    launcher.container.nbrs_pitch = sizes.neighbors_pitch;
    launcher.container.coords_pitch = sizes.coords_pitch;
    launcher.container.snaps_pitch = sizes.snaps_pitch;
    launcher.container.num_border_nodes = sizes.border_nodes;
    launcher.container.num_interior_nodes = connect.getLocalSize() - sizes.border_nodes;

    launcher.container.pack_buf = comm_manager.send_buf_device.get();
    launcher.container.unpack_buf = comm_manager.recv_buf_device.get();
    launcher.container.pack_inds = comm_manager.pack_inds.get();
    launcher.container.unpack_inds = comm_manager.unpack_inds.get();
    launcher.container.pack_sz = static_cast<unsigned int>(comm_manager.send_buf_host.size());
    launcher.container.unpack_sz = static_cast<unsigned int>(comm_manager.recv_buf_host.size());

    const auto dyn_offs_lu = Model_m::makeDynamicOffsetIndLookupTable();
    std::copy(dyn_offs_lu.begin(), dyn_offs_lu.end(), launcher.container.dynamic_offset_lookup_table);
    launcher.container.stencil_offset = Model_m::stencil_offsets;
    launcher.container.stencil_size = Model_m::stencil_sizes;
}

int ArbLattice::fullLatticePos(double pos) const {
    const auto retval = std::lround(pos / connect.grid_size - .5);
    assert(retval <= std::numeric_limits<int>::max() && retval >= std::numeric_limits<int>::min());
    return static_cast<int>(retval);
}

lbRegion ArbLattice::getLocalBoundingBox() const {
    const auto local_sz = connect.getLocalSize();
    const Span x(connect.coords.get(), local_sz), y(std::next(connect.coords.get(), local_sz), local_sz),
        z(std::next(connect.coords.get(), 2 * local_sz), local_sz);
    const auto [minx_it, maxx_it] = std::minmax_element(x.begin(), x.end());
    const auto [miny_it, maxy_it] = std::minmax_element(y.begin(), y.end());
    const auto [minz_it, maxz_it] = std::minmax_element(z.begin(), z.end());
    const int x_min = fullLatticePos(*minx_it), x_max = fullLatticePos(*maxx_it), y_min = fullLatticePos(*miny_it), y_max = fullLatticePos(*maxy_it),
              z_min = fullLatticePos(*minz_it), z_max = fullLatticePos(*maxz_it);
    return lbRegion(x_min, y_min, z_min, x_max - x_min + 1, y_max - y_min + 1, z_max - z_min + 1);
}

ArbLattice::ArbVTUGeom ArbLattice::makeVTUGeom() const {
    using Index = std::int64_t;
    // Bounding box for node-encapsulating cubes is larger by 1 (in each direction) than that of the nodes themselves
    const Index nx = local_bounding_box.nx + 1, ny = local_bounding_box.ny + 1, nz = local_bounding_box.nz + 1;
    const Index sx = local_bounding_box.dx, sy = local_bounding_box.dy, sz = local_bounding_box.dz;
    const auto lin_pos_bb = [&](Index x, Index y, Index z) { return x + (y + z * ny) * nx; };
    const auto get_bb_verts = [&](unsigned node) {
        const double x = connect.coord(0, node), y = connect.coord(1, node), z = connect.coord(2, node);
        const int posx = fullLatticePos(x), posy = fullLatticePos(y), posz = fullLatticePos(z);
        static constexpr std::array offsets = {std::array{0, 0, 0},
                                               std::array{1, 0, 0},
                                               std::array{1, 1, 0},
                                               std::array{0, 1, 0},
                                               std::array{0, 0, 1},
                                               std::array{1, 0, 1},
                                               std::array{1, 1, 1},
                                               std::array{0, 1, 1}};  // We need a specific ordering to agree with the vtu spec
        std::array<Index, 8> retval{};
        std::transform(offsets.begin(), offsets.end(), retval.begin(), [&](const auto& ofs) {
            const auto [dx, dy, dz] = ofs;
            return lin_pos_bb(posx - sx + dx, posy - sy + dy, posz - sz + dz);
        });
        return retval;
    };
    const auto full_to_red_map = std::invoke([&] {  // Map from full bounding box to reduced space
        std::unordered_map<Index, unsigned> retval;
        for (unsigned node = 0; node != connect.getLocalSize(); ++node) {
            const auto verts = get_bb_verts(node);
            for (auto v : verts) retval.emplace(v, 0);
        }
        unsigned red_ind = 0;
        for (auto& [_, i] : retval) i = red_ind++;
        return retval;
    });

    ArbVTUGeom retval{connect.getLocalSize(),
                      full_to_red_map.size(),
                      std::make_unique<double[]>(full_to_red_map.size() * 3),
                      std::make_unique<unsigned[]>(connect.getLocalSize() * 8)};
    // Iterating across the entire bounding box is a bit hairy, but saves memory compared to the alternative (and we only do it once)
    for (Index vx = sx; vx != nx + sx; ++vx)
        for (Index vy = sy; vy != ny + sy; ++vy)
            for (Index vz = sz; vz != nz + sz; ++vz) {
                const auto lin_ind = lin_pos_bb(vx - sx, vy - sy, vz - sz);
                if (const auto iter = full_to_red_map.find(lin_ind); iter != full_to_red_map.end()) {
                    const auto red_ind = iter->second;
                    retval.coords[red_ind * 3] = static_cast<double>(vx) * connect.grid_size;
                    retval.coords[red_ind * 3 + 1] = static_cast<double>(vy) * connect.grid_size;
                    retval.coords[red_ind * 3 + 2] = static_cast<double>(vz) * connect.grid_size;
                }
            }
    for (unsigned node = 0; node != connect.getLocalSize(); ++node) {
        const auto verts = get_bb_verts(node);
        const auto node_permuted = local_permutation[node];
        std::transform(verts.begin(), verts.end(), std::next(retval.verts.get(), node_permuted * verts.size()), [&](Index v) { return full_to_red_map.at(v); });
    }
    return retval;
}

storage_t* ArbLattice::getSnapPtr(int snap_ind) {
    return std::next(snaps_device.get(), sizes.snaps_pitch * NF * snap_ind);
}

#ifdef ADJOINT
storage_t* ArbLattice::getAdjointSnapPtr(int snap_ind) {
    return std::next(snaps_device.get(), sizes.snaps_pitch * NF * (sizes.snaps - 2 + snap_ind));
}
#endif

void ArbLattice::SetFirstTabs(int tab_in, int tab_out) {
    setSnapIn(tab_in);
    setSnapOut(tab_out);
}




std::vector<big_flag_t> ArbLattice::getFlags() const { throw std::runtime_error{"UNIMPLEMENTED"}; return {}; };
std::vector<real_t> ArbLattice::getField(const Model::Field& f) { throw std::runtime_error{"UNIMPLEMENTED"}; return {}; };
std::vector<real_t> ArbLattice::getFieldAdj(const Model::Field& f) { throw std::runtime_error{"UNIMPLEMENTED"}; return {}; };
void ArbLattice::setFlags(const std::vector<big_flag_t>& x) { throw std::runtime_error{"UNIMPLEMENTED"}; return; };
void ArbLattice::setField(const Model::Field& f, const std::vector<real_t>& x) { throw std::runtime_error{"UNIMPLEMENTED"}; return; };
void ArbLattice::setFieldAdjZero(const Model::Field& f) { throw std::runtime_error{"UNIMPLEMENTED"}; return; };


std::vector<real_t> ArbLattice::getQuantity(const Model::Quantity& q, real_t scale) {
    size_t size = getLocalSize();
    int comp = q.getComp();
    std::vector<real_t> ret(size*comp);
    setSnapIn(Snap);
#ifdef ADJOINT
    setAdjSnapIn(aSnap);
#endif
    launcher.getQuantity(q.id, ret.data(), scale, data);
    return ret;
}

std::vector<real_t> ArbLattice::getCoord(const Model::Coord& d, real_t scale) {
    size_t size = getLocalSize();
    std::vector<real_t> ret(size);
    for (size_t i = 0; i < size; ++i) {
        size_t j = local_permutation[i];
        ret[j] = connect.coord(d.id, i)*scale;
    }
    return ret;
}

#include <iostream>

void ArbLattice::initCommManager() {
    if (mpitools::MPI_Size(comm) == 1) return;
    int rank = mpitools::MPI_Rank(comm);
    const auto& field_table = Model_m::field_streaming_table;
    using NodeFieldP = std::array<size_t, 2>;              // Node + field index
    std::map<int, std::vector<NodeFieldP>> needed_fields;  // in_nbrs to required N-F pairs, **we need it to be sorted**
    for (size_t node = 0; node != connect.getLocalSize(); ++node) {
        for (size_t q = 0; q != Q; ++q) {
            const auto nbr = connect.neighbor(q, node);
            if (nbr != -1 && connect.isGhost(nbr)) {
                const int owner = std::distance(global_node_dist.cbegin(), std::upper_bound(global_node_dist.cbegin(), global_node_dist.cend(), nbr)) - 1;
                auto& owner_set = needed_fields[owner];
                for (size_t f = 0; f != NF; ++f)
                    if (field_table.at(f).at(q)) owner_set.push_back(NodeFieldP{static_cast<size_t>(nbr), f});
            }
        }
    }
    for (auto& [id, vec] : needed_fields) {
        std::sort(vec.begin(), vec.end());
        vec.erase(std::unique(vec.begin(), vec.end()), vec.end());
        comm_manager.in_nbrs.emplace_back(id, vec.size());
    }
    const size_t recv_buf_size =
        std::transform_reduce(comm_manager.in_nbrs.cbegin(), comm_manager.in_nbrs.cend(), size_t{0}, std::plus{}, [](auto p) { return p.second; });
    comm_manager.recv_buf_host = std::pmr::vector<storage_t>(recv_buf_size, &global_pinned_resource);
    comm_manager.recv_buf_device = cudaMakeUnique<storage_t>(recv_buf_size);
    comm_manager.unpack_inds = cudaMakeUnique<size_t>(recv_buf_size);
    std::pmr::vector<size_t> unpack_inds_host(recv_buf_size, &global_pinned_resource);
    auto unpack_ind_iter = unpack_inds_host.begin();
    for (const auto& [id, set] : needed_fields) {
        unpack_ind_iter = std::transform(set.begin(), set.end(), unpack_ind_iter, [&](NodeFieldP nfp) {
            const auto [node, field] = nfp;
            return lookupLocalGhostIndex(node) + field * sizes.snaps_pitch;
        });
    }
    assert(unpack_ind_iter == unpack_inds_host.end());
    CudaMemcpyAsync(comm_manager.unpack_inds.get(), unpack_inds_host.data(), unpack_inds_host.size() * sizeof(size_t), CudaMemcpyHostToDevice, inStream);

    std::vector<size_t> comm_sizes_in(mpitools::MPI_Size(comm));
    std::vector<size_t> comm_sizes(mpitools::MPI_Size(comm));
    for (const auto& [id, set] : needed_fields) comm_sizes_in[id] = set.size();
    MPI_Alltoall(comm_sizes_in.data(), 1, mpitools::getMPIType<size_t>(), comm_sizes.data(), 1, mpitools::getMPIType<size_t>(), comm);
    int out_id = 0;
    for (auto sz : comm_sizes) {
        if (sz != 0) comm_manager.out_nbrs.emplace_back(out_id, sz);
        ++out_id;
    }
    size_t send_buf_size =
        std::transform_reduce(comm_manager.out_nbrs.cbegin(), comm_manager.out_nbrs.cend(), size_t{0}, std::plus{}, [](auto p) { return p.second; });
    comm_manager.send_buf_host = std::pmr::vector<storage_t>(send_buf_size, &global_pinned_resource);
    comm_manager.send_buf_device = cudaMakeUnique<storage_t>(send_buf_size);
    comm_manager.pack_inds = cudaMakeUnique<size_t>(send_buf_size);

    std::map<int, std::vector<NodeFieldP>> requested_fields;
    for (const auto& [id, sz] : comm_manager.out_nbrs) {
        auto& rf = requested_fields[id];
        rf.resize(sz);
    }
    std::vector<MPI_Request> reqs;
    reqs.reserve(requested_fields.size() + needed_fields.size());
    for (auto& [id, rf] : requested_fields) MPI_Irecv(rf.data(), rf.size() * 2, mpitools::getMPIType<size_t>(), id, 0, comm, &reqs.emplace_back());
    for (const auto& [id, nf] : needed_fields) MPI_Isend(nf.data(), nf.size() * 2, mpitools::getMPIType<size_t>(), id, 0, comm, &reqs.emplace_back());
    MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
    std::pmr::vector<size_t> pack_inds_host(send_buf_size, &global_pinned_resource);
    auto pack_ind_iter = pack_inds_host.begin();
    for (const auto& [id, nfps] : requested_fields) {
        pack_ind_iter = std::transform(nfps.begin(), nfps.end(), pack_ind_iter, [&](const NodeFieldP nfp) {
            const auto [node, field] = nfp;
            assert(node >= connect.chunk_begin);
            assert(node < connect.chunk_end);
            const auto lid = local_permutation.at(node - connect.chunk_begin);
            assert(lid < sizes.border_nodes);
            return lid + field * sizes.snaps_pitch;
        });
    }
    assert(pack_ind_iter == pack_inds_host.end());
    CudaMemcpyAsync(comm_manager.pack_inds.get(), pack_inds_host.data(), pack_inds_host.size() * sizeof(size_t), CudaMemcpyHostToDevice, inStream);
    CudaStreamSynchronize(inStream);

    if (debug_name.size() != 0) {
        printf("rank %d snaps_pitch %ld\n", rank, sizes.snaps_pitch);
        std::string filename;
        size_t i;
        FILE* f;
        filename = formatAsString("%s_P%02d_pack.csv", debug_name, rank);
        f = fopen(filename.c_str(), "w");
        fprintf(f, "rank,id,globalIdx,field,idx\n");
        i = 0;
        for (const auto& [id, nfps] : requested_fields) {
            for (const auto& [node, field] : nfps) {
                assert(i < pack_inds_host.size());
                const auto& idx = pack_inds_host[i];
                fprintf(f, "%d,%d,%ld,%ld,%ld\n", rank, id, node, field, idx);
                i++;
            }
        }
        assert(i == pack_inds_host.size());
        fclose(f);
        filename = formatAsString("%s_P%02d_unpack.csv", debug_name, rank);
        f = fopen(filename.c_str(), "w");
        fprintf(f, "rank,id,globalIdx,field,idx\n");
        i = 0;
        for (const auto& [id, nfps] : needed_fields) {
            for (const auto& [node, field] : nfps) {
                assert(i < unpack_inds_host.size());
                const auto& idx = unpack_inds_host[i];
                fprintf(f, "%d,%d,%ld,%ld,%ld\n", rank, id, node, field, idx);
                i++;
            }
        }
        assert(i == unpack_inds_host.size());
        fclose(f);
    }
}

void ArbLattice::communicateBorder() {
    std::vector<MPI_Request> reqs(comm_manager.in_nbrs.size() + comm_manager.out_nbrs.size(), MPI_REQUEST_NULL);
    auto get_req = [&reqs, i = 0]() mutable { return &reqs[i++]; };
    size_t offset = 0;
    for (const auto& [id, sz] : comm_manager.in_nbrs) {
        MPI_Irecv(std::next(comm_manager.recv_buf_host.data(), offset), sz, mpitools::getMPIType<storage_t>(), id, 0, comm, get_req());
        offset += sz;
    }
    offset = 0;
    for (const auto& [id, sz] : comm_manager.out_nbrs) {
        MPI_Isend(std::next(comm_manager.send_buf_host.data(), offset), sz, mpitools::getMPIType<storage_t>(), id, 0, comm, get_req());
        offset += sz;
    }
    MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
}

void ArbLattice::MPIStream_A() {
    if (mpitools::MPI_Size(comm) == 1) return;
    launcher.pack(outStream);
    CudaMemcpyAsync(comm_manager.send_buf_host.data(),
                    comm_manager.send_buf_device.get(),
                    comm_manager.send_buf_host.size() * sizeof(storage_t),
                    CudaMemcpyDeviceToHost,
                    outStream);
}

void ArbLattice::MPIStream_B() {
    if (mpitools::MPI_Size(comm) == 1) return;
    CudaStreamSynchronize(outStream);
    communicateBorder();
    CudaMemcpyAsync(comm_manager.recv_buf_device.get(),
                    comm_manager.recv_buf_host.data(),
                    comm_manager.recv_buf_host.size() * sizeof(storage_t),
                    CudaMemcpyHostToDevice,
                    inStream);
    launcher.unpack(inStream);
    CudaStreamSynchronize(inStream);
}

static int saveImpl(const std::string& filename, const storage_t* device_ptr, size_t size) {
    std::pmr::vector<storage_t> tab(size);
    CudaMemcpy(tab.data(), device_ptr, size * sizeof(storage_t), CudaMemcpyDeviceToHost);
    auto file = fopen(filename.c_str(), "wb");
    if (!file) {
        const auto err_msg = std::string("Failed to open ") + filename + " for writing";
        ERROR(err_msg.c_str());
        return EXIT_FAILURE;
    }
    const auto n_written = fwrite(tab.data(), sizeof(storage_t), size, file);
    fclose(file);
    if (n_written != size) {
        const auto err_msg = std::string("Error writing to ") + filename;
        ERROR(err_msg.c_str());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

static int loadImpl(const std::string& filename, storage_t* device_ptr, size_t size) {
    auto file = fopen(filename.c_str(), "rb");
    if (!file) {
        const auto err_msg = std::string("Failed to open ") + filename + " for reading";
        ERROR(err_msg.c_str());
        return EXIT_FAILURE;
    }
    std::pmr::vector<storage_t> tab(size);
    const auto n_read = fread(tab.data(), sizeof(storage_t), size, file);
    fclose(file);
    if (n_read != size) {
        const auto err_msg = std::string("Error reading from ") + filename;
        ERROR(err_msg.c_str());
        return EXIT_FAILURE;
    }
    CudaMemcpy(device_ptr, tab.data(), size * sizeof(storage_t), CudaMemcpyHostToDevice);
    return EXIT_SUCCESS;
}
void ArbLattice::savePrimal(const std::string& filename, int snap_ind) const {
    if (saveImpl(filename, getSnapPtr(snap_ind), sizes.snaps_pitch * NF)) throw std::runtime_error{"savePrimal failed"};
}

int ArbLattice::loadPrimal(const std::string& filename, int snap_ind) {
    return loadImpl(filename, getSnapPtr(snap_ind), sizes.snaps_pitch * NF);
}
#ifdef ADJOINT
int ArbLattice::loadAdj(const std::string& filename, int asnap_ind) {
    throw std::runtime_error{"UNIMPLEMENTED"};
    return -1;
}
void ArbLattice::saveAdj(const std::string& filename, int asnap_ind) const {
    throw std::runtime_error{"UNIMPLEMENTED"};
}
#endif
void ArbLattice::clearAdjoint() {
#ifdef ADJOINT
    debug1("Clearing adjoint\n");
    aSnaps[0].Clear(getLocalRegion().nx, getLocalRegion().ny, getLocalRegion().nz);
    aSnaps[1].Clear(getLocalRegion().nx, getLocalRegion().ny, getLocalRegion().nz);
#endif
    zSet.ClearGrad();
}
/// TODO section end

void ArbLattice::debugDumpConnect(const std::string& name) const {
    if (debug_name.size() != 0) connect.dump(formatAsString("%s_P%02d_%s.csv", debug_name, mpitools::MPI_Rank(comm), name));
}

void ArbLattice::debugDumpVTU() const {
    if (debug_name.size() != 0) {
        std::string filename;
        size_t i;
        FILE* f;

        const int rank = mpitools::MPI_Rank(comm);
        filename = formatAsString("%s_P%02d_loc_perm.csv", debug_name, rank);
        f = fopen(filename.c_str(), "w");
        fprintf(f, "rank,globalIdx,idx\n");
        i = connect.chunk_begin;
        for (const auto& idx : local_permutation) {
            fprintf(f, "%d,%ld,%d\n", rank, i, idx);
            i++;
        }
        assert(i == connect.chunk_end);
        i = getLocalSize();
        for (const auto& gidx : ghost_nodes) {
            fprintf(f, "%d,%ld,%ld\n", rank, gidx, i);
            i++;
        }
        fprintf(f, "%d,%ld,%ld\n", rank, (long int)-1, i);
        i++;
        printf("i:%ld snaps_pitch: %ld\n", i, sizes.snaps_pitch);
        fflush(stdout);
        assert(i <= sizes.snaps_pitch);
        fclose(f);

        filename = formatAsString("%s_P%02d.vtu", debug_name, rank);
        const auto& [num_cells, num_points, coords, verts] = getVTUGeom();
        VtkFileOut vtu_file(filename, num_cells, num_points, coords.get(), verts.get(), MPMD.local, true, false);
        {
            std::vector<size_t> tab1(getLocalSize());
            std::vector<int> tab2(getLocalSize());
            std::vector<size_t> tab3(getLocalSize());
            for (size_t node = 0; node != connect.getLocalSize(); ++node) {
                auto i = local_permutation.at(node);
                tab1[i] = node + connect.chunk_begin;
                tab2[i] = rank;
                tab3[i] = connect.og_index[node];
            }
            vtu_file.writeField("globalId", tab1.data());
            vtu_file.writeField("globalIdRank", tab2.data());
            vtu_file.writeField("globalIdOg", tab3.data());
        }
        {
            std::vector<signed long int> tab1(getLocalSize() * Q);
            std::vector<int> tab2(getLocalSize() * Q);
            for (size_t node = 0; node != connect.getLocalSize(); ++node) {
                auto i = local_permutation.at(node);
                for (size_t q = 0; q != Q; ++q) {
                    const auto nbr = connect.neighbor(q, node);
                    tab1[i * Q + q] = nbr;
                    const int owner = std::distance(global_node_dist.cbegin(), std::upper_bound(global_node_dist.cbegin(), global_node_dist.cend(), nbr)) - 1;
                    tab2[i * Q + q] = owner;
                }
            }
            vtu_file.writeField("neighbour", tab1.data(), Q);
            vtu_file.writeField("neighbourRank", tab2.data(), Q);
        }
        vtu_file.writeFooters();
    }
}
