#ifndef ARBLATTICE_HPP
#define ARBLATTICE_HPP

#include <map>
#include <string>

#include "ArbConnectivity.hpp"
#include "LatticeBase.hpp"
#include "Lists.h"

class ArbLattice : public LatticeBase {
   public:
    static constexpr size_t Q = Model_m::offset_directions.size();

   private:
    ArbLatticeConnectivity connect;
    size_t num_snaps;

   public:
    ArbLattice(size_t num_snaps_, const UnitEnv& units, const std::map<std::string, int>& zone_map, const std::string& cxn_path, MPI_Comm comm) : LatticeBase(ZONESETTINGS, ZONE_MAX, units), num_snaps(num_snaps_) { readFromCxn(zone_map, cxn_path, comm); }

    size_t getLocalSize() const final { return connect.chunk_end - connect.chunk_begin; }
    size_t getGlobalSize() const final { return connect.num_nodes_global; }

    void Iterate(int, int) final {}
    void IterateTill(int, int) final {}

   private:
    std::unordered_map<std::string, int> makeGroupZoneMap(const std::map<std::string, int>& zone_map) const;
    void readFromCxn(const std::map<std::string, int>& zone_map, const std::string& cxn_path, MPI_Comm comm);
};

#endif  // ARBLATTICE_HPP
