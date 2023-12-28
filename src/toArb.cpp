#include "toArb.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <unordered_map>
#include <vector>

static long linPos(long x, long y, long z, long nx, long ny) {
    return x + nx * y + nx * ny * z;
}

// i mod m, assuming i is greater than -m and less than 2*m
static long fastModSingleWrap(long i, long m) {
    if (i < 0) return i + m;
    if (i >= m) return i - m;
    return i;
}

static long linPosBoundschecked(long x, long y, long z, long nx, long ny, long nz) {
    return linPos(fastModSingleWrap(x, nx), fastModSingleWrap(y, ny), fastModSingleWrap(z, nz), nx, ny);
}

// Mark the lattice nodes which are in the bulk region
static auto makeBulkBmp(const Geometry& geo, big_flag_t bulk_mask, big_flag_t bulk_flag) -> std::vector<bool> {
    const auto nx = geo.totalregion.nx, ny = geo.totalregion.ny, nz = geo.totalregion.nz;
    std::vector<bool> retval(nx * ny * nz);
    for (long z = 0; z < nz; z++)
        for (long y = 0; y < ny; y++)
            for (long x = 0; x < nx; x++)
                if ((geo.geom[geo.region.offset(x, y, z)] & bulk_mask) == bulk_flag) {
                    const auto lin_pos = linPos(x, y, z, nx, ny);
                    retval[lin_pos] = true;
                }
    return retval;
}

// Map from full Cartesian lattice linear index to arbitrary lattice index
static auto makeArbLatticeIndexMap(const lbRegion& region, const std::vector<bool>& bulk_bmp) -> std::unordered_map<long, long> {
    const long nx = region.nx, ny = region.ny, nz = region.nz;
    const auto is_void = [&](long x, long y, long z, long lin_pos) {
        if (!bulk_bmp[lin_pos]) return false;  // non-bulk nodes always stay
        return std::all_of(Model_m::offset_directions.begin(), Model_m::offset_directions.end(), [&](const auto& ofs_dir) -> bool {
            const auto [dx, dy, dz] = ofs_dir;
            const auto lin_pos_offset = linPosBoundschecked(x + dx, y + dy, z + dz, nx, ny, nz);
            return bulk_bmp[lin_pos_offset];
        });
    };
    std::unordered_map<long, long> retval;
    retval.max_load_factor(.5);
    for (long index = 0, lin_pos = 0, z = 0; z < nz; ++z)
        for (long y = 0; y < ny; ++y)
            for (long x = 0; x < nx; ++x) {
                if (!is_void(x, y, z, lin_pos)) retval.emplace(lin_pos, index++);
                ++lin_pos;
            }
    return retval;
}

static int writeArbLatticeHeader(std::fstream& file, size_t n_nodes, double grid_size, const Model& model, const std::map<std::string, int>& zone_map) {
    file << "OFFSET_DIRECTIONS " << Model_m::offset_directions.size() << '\n';
    for (const auto [x, y, z] : Model_m::offset_directions) file << x << ' ' << y << ' ' << z << '\n';
    file << "GRID_SIZE " << grid_size << '\n';
    file << "NODE_LABELS " << model.nodetypeflags.size() + zone_map.size() << '\n';
    for (const auto& ntf : model.nodetypeflags) file << ntf.name << '\n';
    for (const auto& [name, zf] : zone_map) file << "_Z_" << name << '\n';
    file << "NODES " << n_nodes << '\n';
    return file.good() ? EXIT_SUCCESS : EXIT_FAILURE;
}

static int writeArbLatticeNodes(const Geometry& geo,
                                const Model& model,
                                const std::map<std::string, int>& zone_map,
                                const std::unordered_map<long, long>& lin_to_arb_index_map,
                                const std::vector<bool>& bulk_bmp,
                                std::fstream& file,
                                double spacing) {
    const long nx = geo.totalregion.nx, ny = geo.totalregion.ny, nz = geo.totalregion.nz;
    const auto get_nbr_id = [&](long my_pos, long nbr_pos) -> long {
        if (my_pos != nbr_pos && bulk_bmp[my_pos] && bulk_bmp[nbr_pos]) return -1;  // ignore edges between bulk nodes
        const auto nbr_it = lin_to_arb_index_map.find(nbr_pos);
        return nbr_it != lin_to_arb_index_map.end() ? nbr_it->second : -1;
    };
    for (long lin_pos = 0, z = 0; z < nz; z++)
        for (long y = 0; y < ny; y++)
            for (long x = 0; x < nx; x++) {
                const auto current_lin_pos = lin_pos++;
                if (lin_to_arb_index_map.find(current_lin_pos) == lin_to_arb_index_map.end()) continue;  // void node

                // Coordinates
                const double x_coord = (static_cast<double>(x) + .5) * spacing;
                const double y_coord = (static_cast<double>(y) + .5) * spacing;
                const double z_coord = (static_cast<double>(z) + .5) * spacing;
                file << x_coord << ' ' << y_coord << ' ' << z_coord << ' ';

                // Neighbors
                for (const auto [dx, dy, dz] : Model_m::offset_directions) {
                    const auto lin_pos_offset = linPosBoundschecked(x + dx, y + dy, z + dz, nx, ny, nz);
                    const auto nbr_it = lin_to_arb_index_map.find(lin_pos_offset);
                    file << get_nbr_id(current_lin_pos, lin_pos_offset) << ' ';
                }

                // Groups & zones
                const auto flag = geo.geom[geo.region.offset(x, y, z)];
                const int zone_flag = (flag & model.settingzones.flag) >> model.settingzones.shift;
                const size_t n_groups = std::count_if(model.nodetypeflags.cbegin(), model.nodetypeflags.cend(), [flag](const auto& ntf) {
                    return ntf.flag != 0 && (flag & ntf.group_flag) == ntf.flag;
                });
                const size_t n_zones = zone_flag == 0 ? 0 : std::count_if(zone_map.begin(), zone_map.end(), [zone_flag](const auto& zone_map_entry) {
                    return zone_map_entry.second == zone_flag;
                });
                file << n_groups + n_zones << ' ';
                size_t gz_ind = 0;
                for (const auto& ntf : model.nodetypeflags) {
                    if (ntf.flag != 0 && (flag & ntf.group_flag) == ntf.flag) file << gz_ind << ' ';
                    ++gz_ind;
                }
                if (zone_flag != 0)
                    for (const auto& [name, zf] : zone_map) {
                        if (zone_flag == zf) file << gz_ind << ' ';
                        ++gz_ind;
                    }
                file << '\n';
                if (!file.good()) break;  // Fail early
            }

    if (!file.good()) ERROR("I/O error while writing .cxn file");
    return file.good() ? EXIT_SUCCESS : EXIT_FAILURE;
}

static int writeArbLattice(const Geometry& geo,
                           const Model& model,
                           const std::map<std::string, int>& zone_map,
                           const std::unordered_map<long, long>& lin_to_arb_index_map,
                           const std::vector<bool>& bulk_bmp,
                           const std::string& filename,
                           double spacing) {
    std::fstream file(filename, std::ios_base::out);
    if (!file.good()) {
        ERROR("Failed to open .cxn file for writing");
        return EXIT_FAILURE;
    }
    if (writeArbLatticeHeader(file, lin_to_arb_index_map.size(), spacing, model, zone_map)) return EXIT_FAILURE;
    return writeArbLatticeNodes(geo, model, zone_map, lin_to_arb_index_map, bulk_bmp, file, spacing);
}

static int writeArbXml(const Solver& solver, const Geometry& geo, const Model& model, const std::string& cxn_path) {
    pugi::xml_document restartfile;
    for (auto n = solver.configfile.first_child(); n; n = n.next_sibling()) restartfile.append_copy(n);
    pugi::xml_node n0 = restartfile.child("CLBConfig");
    pugi::xml_node n1 = n0.child("Geometry");
    if (!n1) {
        ERROR("No geometry node in xml - this should not happen");
        return EXIT_FAILURE;
    }

    auto attr = n0.attribute("toArb");
    if (attr) n0.remove_attribute(attr);
    attr = n0.attribute("remove_bulk");
    if (attr) n0.remove_attribute(attr);
    pugi::xml_node n2 = n0.insert_child_before("ArbitraryLattice", n1);
    n0.remove_child(n1);
    n2.append_attribute("file").set_value(cxn_path.c_str());
    for (const auto& ntf : model.nodetypeflags) {
        if (ntf.flag == 0) continue;
        pugi::xml_node n3 = n2.append_child(ntf.name.c_str());
        n3.append_attribute("group").set_value(ntf.name.c_str());
    }
    for (const auto& [name, zs] : solver.setting_zones) {
        if (zs == 0) continue;
        auto n3 = n2.append_child("None");
        const auto group_str = "_Z_" + name;
        n3.append_attribute("group").set_value(group_str.c_str());
        n3.append_attribute("name").set_value(name.c_str());
    }

    const auto filename = solver.outGlobalFile("ARB", ".xml");
    output("Writing modified xml config to %s...", filename.c_str());
    return restartfile.save_file(filename.c_str()) ? EXIT_SUCCESS : EXIT_FAILURE;
}

static long liRegionSize(const lbRegion& region) {
    const long nx = region.nx, ny = region.ny, nz = region.nz;
    return nx * ny * nz;
}

int toArbitrary(const Solver& solver, const Geometry& geo, const Model& model) {
    output("Converting Cartesian geometry to arbitrary grid format...");
    auto attr = solver.configfile.child("CLBConfig").attribute("remove_bulk");
    const char* remove_bulk = attr ? attr.value() : nullptr;
    big_flag_t bulk_flag = -1, bulk_mask = 0;  // Match nothing by default
    if (remove_bulk) {
        const auto& ntf = model.nodetypeflags.by_name(remove_bulk);
        bulk_flag = ntf.flag;
        bulk_mask = ntf.group_flag;
    }
    const auto bulk_bmp = makeBulkBmp(geo, bulk_mask, bulk_flag);
    const auto id_map = makeArbLatticeIndexMap(geo.totalregion, bulk_bmp);
    output("Interior size: %lu / %li", id_map.size(), liRegionSize(geo.totalregion));
    const auto filename = solver.outGlobalFile("ARB", ".cxn");
    const double spacing = 1 / solver.units.alt("m");
    output("Writing arbitrary lattice data to %s...", filename.c_str());
    if (writeArbLattice(geo, model, solver.setting_zones, id_map, bulk_bmp, filename, spacing)) return EXIT_FAILURE;
    return writeArbXml(solver, geo, model, filename);
}
