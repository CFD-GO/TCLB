#ifndef ARBCONNECTIVITY_HPP
#define ARBCONNECTIVITY_HPP

#include <cstdint>

using arb_local_id = std::uint32;
using arb_global_id = std::uint64;

template <size_t Q>  // number of connectivity directions
struct ArbConnectivity {
    std::vector<std::array<arb_local_id, Q>> neighbors;
    std::vector<arb_global_id> global_ids;
};

#endif  // ARBCONNECTIVITY_HPP
