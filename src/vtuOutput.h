#ifndef VTUOUTPUT_H
#define VTUOUTPUT_H

#include <mpi.h>

#include "cross.h"
#include "types.h"
#include "utils.h"

namespace detail {
template <typename T>
struct VTKTypeTraits;

#define DEF_VTK_TYPE_TRAITS(type__, name__, comps__)     \
    template <>                                          \
    struct VTKTypeTraits<type__> {                       \
        static constexpr std::string_view name = name__; \
        static constexpr size_t components = comps__;    \
    }

DEF_VTK_TYPE_TRAITS(char, "Int8", 1);
DEF_VTK_TYPE_TRAITS(short int, "Int16", 1);
DEF_VTK_TYPE_TRAITS(int, "Int32", 1);
DEF_VTK_TYPE_TRAITS(std::int64_t, "Int64", 1);
DEF_VTK_TYPE_TRAITS(unsigned char, "UInt8", 1);
DEF_VTK_TYPE_TRAITS(unsigned short int, "UInt16", 1);
DEF_VTK_TYPE_TRAITS(unsigned int, "UInt32", 1);
DEF_VTK_TYPE_TRAITS(std::uint64_t, "UInt64", 1);
DEF_VTK_TYPE_TRAITS(float, "Float32", 1);
DEF_VTK_TYPE_TRAITS(float2, "Float32", 2);
DEF_VTK_TYPE_TRAITS(float3, "Float32", 3);
DEF_VTK_TYPE_TRAITS(double, "Float64", 1);
DEF_VTK_TYPE_TRAITS(double2, "Float64", 2);
DEF_VTK_TYPE_TRAITS(double3, "Float64", 3);
#ifndef CALC_DOUBLE_PRECISION
DEF_VTK_TYPE_TRAITS(vector_t, "Float32", 3);
#else
DEF_VTK_TYPE_TRAITS(vector_t, "Float64", 3);
#endif
#undef DEF_VTK_TYPE_TRAITS

struct FileCloser {
    void operator()(FILE* f) const noexcept { std::fclose(f); }
};
}  // namespace detail

class VtkFileOut {
    std::unique_ptr<FILE, detail::FileCloser> f, fp;
    std::string name;
    MPI_Comm comm{};
    size_t num_cells{}, num_points{};

    void init();
    void writeHeaders(const double* coords, const unsigned* verts, bool has_scalars, bool has_vectors) const;
    void writePieceInfo() const;
    void writeGeomInfo(const double* coords, const unsigned* verts) const;
    void writeFieldImpl(const std::string& name, const void* data, size_t size, std::string_view vtk_type_name, int components) const;

   public:
    VtkFileOut(std::string name, size_t num_cells, size_t num_points, const double* coords, const unsigned* verts, MPI_Comm comm, bool has_scalars, bool has_vectors);
    void writeFooters() const;

    template <typename T>
    void writeField(const std::string& name, const T* data, size_t comps = 1) const {
        if (comps != 1) assert(detail::VTKTypeTraits<T>::components == 1);  // Passing number of components with a vector/array type makes no sense
        comps = detail::VTKTypeTraits<T>::components * comps;
        writeFieldImpl(name, data, sizeof(T) * comps * num_cells, detail::VTKTypeTraits<T>::name, comps);
    }
};

#endif
