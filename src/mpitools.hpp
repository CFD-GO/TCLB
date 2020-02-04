#ifndef MPITOOLS_H
#define MPITOOLS_H


#include <mpi.h>
#include <string>

namespace mpitools {

inline std::string MPI_Bcast(const std::string& str, int root, MPI_Comm comm) {
        size_t size = str.size();
        ::MPI_Bcast(&size, 1, MPI_UNSIGNED_LONG, root, comm);
        char * buf = new char[size+1];
        strcpy(buf, str.c_str());
        ::MPI_Bcast(buf, size+1, MPI_CHAR, root, comm);
        std::string ret(buf,size);
        delete[] buf;
        return ret;
}

inline std::string MPI_Nodename(MPI_Comm comm) {
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
                MPI_Allreduce(&wrank, &firstrank, 1, MPI_INT, MPI_MIN, comm );
                if (firstrank >= size) break;
        }
        MPI_Comm newcomm;
        MPI_Comm_split(comm, color, rank, &newcomm);
        return newcomm;
}

};
#endif
