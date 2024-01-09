#include "vtkLattice.h"

#include <mpi.h>

#include <cassert>
#include <cstdio>

#include "Global.h"
#include "cross.h"

int vtkWriteLattice(const std::string& filename, CartLattice& lattice, const UnitEnv& units, const name_set& what, const lbRegion& total_output_reg) {
    const lbRegion& local_reg = lattice.getLocalRegion();
    const lbRegion reg = local_reg.intersect(total_output_reg);
    size_t size = reg.size();
    myprint(1,
            -1,
            "Writing region %dx%dx%d + %d,%d,%d (size %d) from %dx%dx%d + %d,%d,%d",
            reg.nx,
            reg.ny,
            reg.nz,
            reg.dx,
            reg.dy,
            reg.dz,
            size,
            local_reg.nx,
            local_reg.ny,
            local_reg.nz,
            local_reg.dx,
            local_reg.dy,
            local_reg.dz);

    vtkFileOut vtkFile(MPMD.local);
    if (vtkFile.Open(filename.c_str())) return -1;
    double spacing = 1 / units.alt("m");
    vtkFile.Init(total_output_reg, reg, "Scalars=\"rho\" Vectors=\"velocity\"", spacing, lattice.px * spacing, lattice.py * spacing, lattice.pz * spacing);

    {
        std::vector<big_flag_t> NodeType = lattice.getFlags(reg);
        if (what.explicitlyIn("flag")) { vtkFile.WriteField("flag", NodeType.data()); }
        std::vector<unsigned char> small(size);
        for (const Model::NodeTypeGroupFlag& it : lattice.model->nodetypegroupflags) {
            if ((what.all && it.isSave) || what.explicitlyIn(it.name)) {
                for (size_t i = 0; i < size; i++) { small[i] = (NodeType[i] & it.flag) >> it.shift; }
                vtkFile.WriteField(it.name.c_str(), small.data());
            }
        }
    }

    for (const Model::Quantity& it : lattice.model->quantities) {
        if (what.in(it.name)) {
            double v = units.alt(it.unit);
            int comp = 1;
            if (it.isVector) comp = 3;
            std::vector<real_t> tmp = lattice.getQuantity(it, reg, 1 / v);
            vtkFile.WriteField(it.name.c_str(), tmp.data(), comp);
        }
    }
    vtkFile.Finish();
    vtkFile.Close();
    return 0;
}

int vtuWriteLattice(const std::string& filename, ArbLattice& lattice, const UnitEnv& units, const name_set& what) {
    try {
        const auto& [num_cells, num_points, coords, verts] = lattice.getVTUGeom();
        const bool has_scalars = std::find_if(lattice.model->quantities.begin(), lattice.model->quantities.end(), [](const auto& q) { return !q.isVector; }) !=
                                 lattice.model->quantities.end();
        const bool has_vectors = std::find_if(lattice.model->quantities.begin(), lattice.model->quantities.end(), [](const auto& q) { return q.isVector; }) !=
                                 lattice.model->quantities.end();
        VtkFileOut vtu_file(filename, num_cells, num_points, coords.get(), verts.get(), MPMD.local, has_scalars, has_vectors);

        const auto node_types_view = lattice.getNodeTypes();
        if (what.explicitlyIn("flag")) { vtu_file.writeField("flag", node_types_view.data()); }
        auto small = std::make_unique<unsigned char[]>(lattice.getLocalSize());
        for (const auto& ntgf : lattice.model->nodetypegroupflags) {
            if ((what.all && ntgf.isSave) || what.explicitlyIn(ntgf.name)) {
                for (size_t i = 0; i < lattice.getLocalSize(); ++i) { small[i] = (node_types_view[i] & ntgf.flag) >> ntgf.shift; }
                vtu_file.writeField(ntgf.name, small.get());
            }
        }
        small.reset();

        for (const auto& quant : lattice.model->quantities) {
            if (what.in(quant.name)) {
                const double v = units.alt(quant.unit);
                const size_t comps = quant.isVector ? 3 : 1;
                auto tmp = lattice.getQuantity(quant, 1 / v);
                vtu_file.writeField(quant.name, tmp.data(), comps);
            }
        }
        vtu_file.writeFooters();
    } catch (const std::exception& e) {
        ERROR(e.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

int binWriteLattice(const std::string& filename, LatticeBase& lattice, const UnitEnv& units) {
    int size = lattice.getLocalSize();
    for (const Model::Quantity& it : lattice.model->quantities) {
        int comp = it.getComp();
        auto tmp = lattice.getQuantity(it, 1);
        const auto fn = formatAsString("%s.%s.bin", filename, it.name);
        FILE* f = fopen(fn.c_str(), "w");
        if (f == NULL) {
            ERROR("Cannot open file: %s\n", fn.c_str());
            return -1;
        }
        fwrite(tmp.data(), sizeof(real_t) * comp, size, f);
        fclose(f);
    }
    return 0;
}

inline int txtWriteElement(FILE* f, float tmp) {
    return fprintf(f, "%.8g", tmp);
}
inline int txtWriteElement(FILE* f, double tmp) {
    return fprintf(f, "%.16lg", tmp);
}
template <typename T>
int txtWriteField(FILE* f, T* tmp, int stop, int n) {
    for (int i = 0; i < n; i++) {
        txtWriteElement(f, tmp[i]);
        if (((i + 1) % stop) == 0) fprintf(f, "\n");
        else
            fprintf(f, " ");
    }
    return 0;
}

int txtWriteLattice(const std::string& filename, LatticeBase& lattice, const UnitEnv& units, const name_set& what, int type) {
    const size_t size = lattice.getLocalSize();
    std::vector<int> shp = lattice.shape();
    int row = 1;
    if (shp.size() > 1) row = shp[0];
    if (D_MPI_RANK == 0) {
        const auto fn = formatAsString("%s_info.txt", filename);
        FILE* f = fopen(fn.c_str(), "w");
        if (f == NULL) {
            ERROR("Cannot open file: %s\n", fn.c_str());
            return -1;
        }
        fprintf(f, "dx: %lg\n", 1 / units.alt("m"));
        fprintf(f, "dt: %lg\n", 1 / units.alt("s"));
        fprintf(f, "dm: %lg\n", 1 / units.alt("kg"));
        fprintf(f, "dT: %lg\n", 1 / units.alt("K"));
        fprintf(f, "size: %ld\n", size);
	fprintf(f, "shape:");
	for (int d : shp) fprintf(f, " %d", d);
	fprintf(f, "\n");
        fclose(f);
    }

    for (const Model::Quantity& it : lattice.model->quantities) {
        if (what.in(it.name)) {
            const auto fn = formatAsString("%s_%s.txt", filename, it.name);
            FILE* f = NULL;
            switch (type) {
                case 0:
                    f = fopen(fn.c_str(), "w");
                    break;
                case 1: {
                    const auto com = formatAsString("gzip > %s.gz", fn);
                    f = popen(com.c_str(), "w");
                    break;
                }
                default:
                    ERROR("Unknown type in txtWriteLattice\n");
            }
            if (f == NULL) {
                ERROR("Cannot open file: %s\n", fn.c_str());
                return -1;
            }
            double v = units.alt(it.unit);
            int comp = it.getComp();
            auto tmp = lattice.getQuantity(it, 1 / v);
            txtWriteField(f, tmp.data(), row*comp, size*comp);
            fclose(f);
        }
    }

    return 0;
}
