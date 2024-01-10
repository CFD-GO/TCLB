#ifndef PARTITIONARBLATTICE_HPP
#define PARTITIONARBLATTICE_HPP

#include <mpi.h>

#include <string>

#include "ArbConnectivity.hpp"

// We need to store some logs here, since we don't have access to fancy printing from within the partitioning utilities (due to conflicting typedefs)
struct PartOutput {
    enum struct MsgType { Notice, Warning, Error };
    struct LoggedEvent {
        MsgType type;
        std::string message;
    };

    std::vector<long> partition_distribution;
    std::vector<LoggedEvent> event_log;
};

///
/// \param lattice arbitrary lattice connectivity info partitioned according to `computeInitialNodeDist`
/// \param comm MPI communicator, result will be distributed among its processes
/// \param dir_wgts offset direction weights (length must be equal to lattice.Q
/// \param self_edge_ind index of the offset direction (0,0,0). If (0,0,0) is not an offset direction, self_edge_ind should be set to any value outside the interval [0, Q)
/// \return distribution (in the ParMETIS sense) of nodes among the processes of comm
PartOutput partitionArbLattice(ArbLatticeConnectivity& lattice, const std::vector<size_t>& dir_wgts, size_t self_edge_ind, MPI_Comm comm);

#endif  // PARTITIONARBLATTICE_HPP
