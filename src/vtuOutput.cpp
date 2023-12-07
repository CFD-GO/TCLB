#include "vtuOutput.h"

#include "mpitools.hpp"

VtkFileOut::VtkFileOut(std::string name_, size_t num_cells_, size_t num_points_, const double* coords, const unsigned* verts, MPI_Comm comm_, bool has_scalars, bool has_vectors) : name(std::move(name_)), comm(comm_), num_cells(num_cells_), num_points(num_points_) {
    init();
    writeHeaders(coords, verts, has_scalars, has_vectors);
}

void VtkFileOut::init() {
    using namespace std::string_literals;
    f.reset(fopen(name.c_str(), "w"));
    if (!f) throw std::runtime_error{"Could not open file: "s + name};
    if (mpitools::MPI_Rank(comm) == 0) {
        const std::string pvtu_path = std::filesystem::path(name).replace_extension("pvtu");
        fp.reset(fopen(pvtu_path.c_str(), "w"));
        if (!fp) throw std::runtime_error{"Could not open file: "s + name};
    }
}

void VtkFileOut::writeHeaders(const double* coords, const unsigned* verts, bool has_scalars, bool has_vectors) const {
    fprintf(f.get(), "<?xml version=\"1.0\"?>\n<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n<UnstructuredGrid>\n");
    fprintf(f.get(), "<Piece NumberOfPoints=\"%lu\" NumberOfCells=\"%lu\">\n", num_points, num_cells);
    fprintf(f.get(), "<PointData>\n</PointData>\n");

    if (fp) fprintf(fp.get(), "<?xml version=\"1.0\"?>\n<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n<PUnstructuredGrid>\n<PPointData>\n</PPointData>\n");
    writePieceInfo();
    writeGeomInfo(coords, verts);

    fprintf(f.get(), "<CellData ");
    if (fp) fprintf(fp.get(), "<PCellData ");
    if (has_scalars) {
        fprintf(f.get(), "Scalars=\"Rho\" ");
        if (fp) fprintf(fp.get(), "Scalars=\"Rho\" ");
    }
    if (has_vectors) {
        fprintf(f.get(), "Vectors=\"Velocity\" ");
        if (fp) fprintf(fp.get(), "Vectors=\"Velocity\" ");
    }
    fprintf(f.get(), ">\n");
    if (fp) fprintf(fp.get(), ">\n");
}

void VtkFileOut::writePieceInfo() const {
    const bool am0 = mpitools::MPI_Rank(comm) == 0;
    std::vector<int> name_sizes(am0 ? mpitools::MPI_Size(comm) : 0);
    const int name_sz = name.size() + 1;
    MPI_Gather(&name_sz, 1, mpitools::getMPIType<int>(), name_sizes.data(), 1, mpitools::getMPIType<int>(), 0, comm);
    auto name_offsets = name_sizes;
    std::exclusive_scan(name_sizes.cbegin(), name_sizes.cend(), name_offsets.begin(), 0);
    std::vector<char> names(am0 ? name_offsets.back() + name_sizes.back() : 0);
    MPI_Gatherv(name.data(), name_sz, mpitools::getMPIType<char>(), names.data(), name_sizes.data(), name_offsets.data(), mpitools::getMPIType<char>(), 0, comm);
    if (am0)
        for (int i = 0; i != mpitools::MPI_Size(comm); ++i) {
            const std::string_view piece_name(std::next(names.data(), name_offsets[i]), name_sizes[i]);
            const std::string piece_fn = std::filesystem::path(piece_name).filename();
            fprintf(fp.get(), "<Piece Source=\"%s\"/>\n", piece_fn.c_str());
        }
}

void VtkFileOut::writeGeomInfo(const double* coords, const unsigned* verts) const {
    // Points
    fprintf(f.get(), "<Points>\n");
    if (fp) fprintf(fp.get(), "<PPoints>\n");
    writeFieldImpl("Position", coords, num_points * 3 * sizeof(double), "Float64", 3);
    fprintf(f.get(), "</Points>\n");
    if (fp) fprintf(fp.get(), "</PPoints>\n");

    // Cells
    constexpr std::uint8_t hex_cell_code = 12;
    const std::vector<std::uint8_t> cell_types(num_cells, hex_cell_code);  // cell types (all hex)
    std::vector<std::uint64_t> cell_offsets(num_cells);                    // cell offsets, because all cells are hex's this is just: 8, 16, 24, ...
    size_t ofs = 0;
    for (auto& o : cell_offsets) {
        ofs += 8;
        o = ofs;
    }
    fprintf(f.get(), "<Cells>\n");
    if (fp) fprintf(fp.get(), "<PCells>\n");
    writeFieldImpl("connectivity", verts, num_cells * sizeof(unsigned) * 8, "UInt32", 1);
    writeField("offsets", cell_offsets.data());
    writeField("types", cell_types.data());
    fprintf(f.get(), "</Cells>\n");
    if (fp) fprintf(fp.get(), "</PCells>\n");
}

void VtkFileOut::writeFooters() const {
    fprintf(f.get(), "</CellData>\n</Piece>\n</UnstructuredGrid>\n</VTKFile>\n");
    if (fp) fprintf(fp.get(), "</PCellData>\n</PUnstructuredGrid>\n</VTKFile>\n");
}

void VtkFileOut::writeFieldImpl(const std::string& name, const void* data, size_t size, std::string_view vtk_type_name, int components) const {
    fprintf(f.get(), "<DataArray type=\"%s\" Name=\"%s\" format=\"binary\" encoding=\"base64\" NumberOfComponents=\"%d\">\n", vtk_type_name.data(), name.c_str(), components);
    fprintB64(f.get(), &size, sizeof(size_t));
    fprintB64(f.get(), data, size);
    fprintf(f.get(), "\n</DataArray>\n");
    if (fp) fprintf(fp.get(), "<PDataArray type=\"%s\" Name=\"%s\" format=\"binary\" encoding=\"base64\" NumberOfComponents=\"%d\" />\n", vtk_type_name.data(), name.c_str(), components);
}