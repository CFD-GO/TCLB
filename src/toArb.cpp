#include "toArb.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <unordered_map>
#include <vector>

static long int linPos(long int x, long int y, long int z, long int nx, long int ny) {
    return x + nx * y + nx * ny * z;
}

// i mod m, assuming i is greater than -m and less than 2*m
static long int fastModSingleWrap(long int i, long int m) {
    if (i < 0) return i + m;
    if (i >= m) return i - m;
    return i;
}

static long int linPosBoundschecked(long int x, long int y, long int z, long int nx, long int ny, long int nz) {
    return linPos(fastModSingleWrap(x, nx), fastModSingleWrap(y, ny), fastModSingleWrap(z, nz), nx, ny);
}

// Mark the lattice nodes which are in the bulk region
static auto makeBulkBmp(const Geometry& geo, big_flag_t bulk_mask, big_flag_t bulk_flag) -> std::vector<bool> {
    const auto nx = geo.totalregion.nx, ny = geo.totalregion.ny, nz = geo.totalregion.nz;
    std::vector<bool> retval(nx * ny * nz);
    for (long int z = 0; z < nz; z++)
        for (long int y = 0; y < ny; y++)
            for (long int x = 0; x < nx; x++)
                if ((geo.geom[geo.region.offset(x, y, z)] & bulk_mask) == bulk_flag) {
                    const auto lin_pos = linPos(x, y, z, nx, ny);
                    retval[lin_pos] = true;
                }
    return retval;
}

// Mark void lattice nodes, i.e., bulk nodes which have all bulk neighbors
static auto makeVoidBmp(const lbRegion& region, const std::vector<bool>& bulk_bmp) -> std::vector<bool> {
    const auto nx = region.nx, ny = region.ny, nz = region.nz;
    std::vector<bool> retval(nx * ny * nz);
    for (long int z = 0; z < nz; z++)
        for (long int y = 0; y < ny; y++)
            for (long int x = 0; x < nx; x++) {
                const auto lin_pos = linPos(x, y, z, nx, ny);
                if (!bulk_bmp[lin_pos]) continue;  // Non-bulk nodes always stay
                retval[lin_pos] = std::all_of(Model_m::offset_directions.begin(), Model_m::offset_directions.end(), [&](const auto& ofs_dir) -> bool {
                    const auto [dx, dy, dz] = ofs_dir;
                    const auto lin_pos_offset = linPosBoundschecked(x + dx, y + dy, z + dz, nx, ny, nz);
                    return bulk_bmp[lin_pos_offset];
                });
            }
    return retval;
}

// Map from full Cartesian lattice linear index to arbitrary lattice index
static auto makeArbLatticeIndexMap(const lbRegion& region, const std::vector<bool>& void_bmp) -> std::unordered_map<long int, long int> {
    const auto nx = region.nx, ny = region.ny, nz = region.nz;
    std::unordered_map<long int, long int> retval(void_bmp.size() - std::count(void_bmp.begin(), void_bmp.end(), true));
    long int index = 0;
    for (long int z = 0; z < nz; z++)
        for (long int y = 0; y < ny; y++)
            for (long int x = 0; x < nx; x++) {
                const auto lin_pos = linPos(x, y, z, nx, ny);
                if (!void_bmp[lin_pos]) retval.emplace(lin_pos, index++);
            }
    return retval;
}

static int writeArbLatticeHeader(std::fstream& file, size_t n_nodes, double underlying_grid_size, const Model& model, const std::map<std::string, int>& zone_map) {
    file << "OFFSET_DIRECTIONS " << Model_m::offset_directions.size() << '\n';
    for (const auto [x, y, z] : Model_m::offset_directions) file << x << ' ' << y << ' ' << z << '\n';
    file << "GRID_SIZE " << underlying_grid_size << '\n';
    file << "NODE_GROUPS " << model.nodetypeflags.size() + zone_map.size() << '\n';
    for (const auto& ntf : model.nodetypeflags) file << ntf.name << '\n';
    for (const auto& [name, zf] : zone_map) file << "_Z_" << name << '\n';
    file << "NODES " << n_nodes << '\n';
    return file.good() ? EXIT_SUCCESS : EXIT_FAILURE;
}

static int writeArbLatticeNodes(const Geometry& geo, const Model& model, const std::map<std::string, int>& zone_map, const std::unordered_map<long int, long int>& lin_to_arb_index_map, const std::vector<bool>& void_bmp, std::fstream& file, double spacing) {
    const auto nx = geo.totalregion.nx, ny = geo.totalregion.ny, nz = geo.totalregion.nz;
    for (long int z = 0; z < nz; z++)
        for (long int y = 0; y < ny; y++)
            for (long int x = 0; x < nx; x++) {
                const auto lin_pos = linPos(x, y, z, nx, ny);
                if (void_bmp[lin_pos]) continue;

                // Coordinates
                const double x_coord = (static_cast<double>(x) + .5) * spacing;
                const double y_coord = (static_cast<double>(y) + .5) * spacing;
                const double z_coord = (static_cast<double>(z) + .5) * spacing;
                file << x_coord << ' ' << y_coord << ' ' << z_coord << ' ';

                // Neighbors
                for (const auto [dx, dy, dz] : Model_m::offset_directions) {
                    const auto lin_pos_offset = linPosBoundschecked(x + dx, y + dy, z + dz, nx, ny, nz);
                    const auto nbr_it = lin_to_arb_index_map.find(lin_pos_offset);
                    file << (nbr_it != lin_to_arb_index_map.end() ? nbr_it->second : -1) << ' ';
                }

                // Groups & zones
                const auto flag = geo.geom[geo.region.offset(x, y, z)];
                const int zone_flag = (flag & model.settingzones.flag) >> model.settingzones.shift;
                const size_t n_groups = std::count_if(model.nodetypeflags.cbegin(), model.nodetypeflags.cend(), [flag](const auto& ntf) { return ntf.flag != 0 && (flag & ntf.group_flag) == ntf.flag; });
                const size_t n_zones = zone_flag == 0 ? 0 : std::count_if(zone_map.begin(), zone_map.end(), [zone_flag](const auto& zone_map_entry) { return zone_map_entry.second == zone_flag; });
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

static int writeArbLattice(const Geometry& geo, const Model& model, const std::map<std::string, int>& zone_map, const std::unordered_map<long int, long int>& lin_to_arb_index_map, const std::vector<bool>& void_bmp, const std::string& filename, double spacing) {
    std::fstream file(filename, std::ios_base::out);
    if (!file.good()) {
        ERROR("Failed to open .cxn file for writing");
        return EXIT_FAILURE;
    }
    if (writeArbLatticeHeader(file, lin_to_arb_index_map.size(), spacing, model, zone_map)) return EXIT_FAILURE;
    if (writeArbLatticeNodes(geo, model, zone_map, lin_to_arb_index_map, void_bmp, file, spacing)) return EXIT_FAILURE;
    return EXIT_SUCCESS;
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
        pugi::xml_node n3 = n2.append_child("Group");
        n3.append_attribute("name").set_value(ntf.name.c_str());
    }
    for (const auto& [name, zs] : solver.setting_zones) {
        if (zs == 0) continue;
        auto n3 = n2.append_child("Zone");
        n3.append_attribute("name").set_value(name.c_str());
    }

    const auto filename = solver.outGlobalFile("ARB", ".xml");
    output("Writing modified xml config to %s...", filename.c_str());
    return restartfile.save_file(filename.c_str()) ? EXIT_SUCCESS : EXIT_FAILURE;
}

static long int liRegionSize(const lbRegion& region) {
    const long int nx = region.nx, ny = region.ny, nz = region.nz;
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
    auto bulk_bmp = makeBulkBmp(geo, bulk_mask, bulk_flag);
    const auto void_bmp = makeVoidBmp(geo.totalregion, bulk_bmp);
    bulk_bmp = {};  // Explicitly free memory for subsequent stages
    const auto id_map = makeArbLatticeIndexMap(geo.totalregion, void_bmp);
    output("Interior size: %lu / %li", id_map.size(), liRegionSize(geo.totalregion));
    const auto filename = solver.outGlobalFile("ARB", ".cxn");
    const double spacing = 1 / solver.units.alt("m");
    output("Writing arbitrary lattice data to %s...", filename.c_str());
    if (writeArbLattice(geo, model, solver.setting_zones, id_map, void_bmp, filename, spacing)) return EXIT_FAILURE;
    return writeArbXml(solver, geo, model, filename);
}
