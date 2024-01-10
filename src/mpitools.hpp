#ifndef MPITOOLS_H
#define MPITOOLS_H

#include <mpi.h>

#include <cstring>
#include <string>
#include <vector>

namespace mpitools {

/// Automatic MPI type deduction
template <typename T>
struct DeduceMPITypeHelper {};

#define SPECIALIZE_MPI_DEDUCTION_HELPER(c_type, mpi_type) \
    template <>                                           \
    struct DeduceMPITypeHelper<c_type> {                  \
        static MPI_Datatype get() {                       \
            return mpi_type;                              \
        }                                                 \
    };

SPECIALIZE_MPI_DEDUCTION_HELPER(char, MPI_CHAR)
SPECIALIZE_MPI_DEDUCTION_HELPER(short, MPI_SHORT)
SPECIALIZE_MPI_DEDUCTION_HELPER(int, MPI_INT)
SPECIALIZE_MPI_DEDUCTION_HELPER(long, MPI_LONG)
SPECIALIZE_MPI_DEDUCTION_HELPER(unsigned char, MPI_UNSIGNED_CHAR)
SPECIALIZE_MPI_DEDUCTION_HELPER(unsigned short, MPI_UNSIGNED_SHORT)
SPECIALIZE_MPI_DEDUCTION_HELPER(unsigned, MPI_UNSIGNED)
SPECIALIZE_MPI_DEDUCTION_HELPER(unsigned long, MPI_UNSIGNED_LONG)
SPECIALIZE_MPI_DEDUCTION_HELPER(long long int, MPI_LONG_LONG_INT)
SPECIALIZE_MPI_DEDUCTION_HELPER(float, MPI_FLOAT)
SPECIALIZE_MPI_DEDUCTION_HELPER(double, MPI_DOUBLE)

#undef SPECIALIZE_MPI_DEDUCTION_HELPER

template <typename T>
MPI_Datatype getMPIType() {
    return DeduceMPITypeHelper<T>::get();
}
//////////////////////////////

template <typename T>
int MPI_Irecv(std::vector<T>& buf, size_t count, int from, int tag, MPI_Comm comm, MPI_Request* req) {
    buf.resize(count);
    return MPI_Irecv(buf.data(), static_cast<int>(count), getMPIType<T>(), from, tag, comm, req);
}

template <typename T>
int MPI_Isend(const std::vector<T>& buf, int to, int tag, MPI_Comm comm, MPI_Request* req) {
    return MPI_Isend(buf.data(), static_cast<int>(buf.size()), getMPIType<T>(), to, tag, comm, req);
}

inline std::string MPI_Bcast(const std::string& str, int root, MPI_Comm comm) {
    size_t size = str.size();
    ::MPI_Bcast(&size, 1, MPI_UNSIGNED_LONG, root, comm);
    char* buf = new char[size + 1];
    strcpy(buf, str.c_str());
    ::MPI_Bcast(buf, static_cast<int>(size + 1), MPI_CHAR, root, comm);
    std::string ret(buf, size);
    delete[] buf;
    return ret;
}

inline std::string MPI_Nodename() {
    int cpname_len;
    char cpname[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(cpname, &cpname_len);
    return std::string(cpname, cpname_len);
}

inline int MPI_Rank(MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    return rank;
}

inline int MPI_Size(MPI_Comm comm) {
    int size;
    MPI_Comm_size(comm, &size);
    return size;
}

inline MPI_Comm MPI_Split(const std::string& str, MPI_Comm comm) {
    int rank = MPI_Rank(comm);
    int size = MPI_Size(comm);
    int wrank = rank;
    int firstrank = 0;
    int color = -1;
    int i = 0;
    while (true) {
        std::string otherstr = MPI_Bcast(str, firstrank, comm);
        if (otherstr == str) {
            wrank = size;
            color = i;
        }
        i++;
        MPI_Allreduce(&wrank, &firstrank, 1, MPI_INT, MPI_MIN, comm);
        if (firstrank >= size) break;
    }
    MPI_Comm newcomm;
    MPI_Comm_split(comm, color, rank, &newcomm);
    return newcomm;
}

};  // namespace mpitools
#endif
