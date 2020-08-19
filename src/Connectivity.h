#ifndef CONNECTIVITY_H
#define CONNECTIVITY_H

#include "unit.h"
#include "Lists.h"


class Connectivity {
    ModelBase* model;
public:
    big_flag_t * geom; ///< Main table of flags/NodeType's
    size_t * connectivity; ///< Main connectivity matrix
    size_t latticeSize; ///< Number of nodes in the arbitrary lattice
    real_t * coords; ///< Table of coordinates of each node
    lbRegion region; ///< Local lattice region
    lbRegion totalregion; ///< Global lattice region
    UnitEnv units; ///< Units object for unit calculations

    Connectivity(const lbRegion& r, const lbRegion& tr, const UnitEnv& units_, ModelBase * model_);
    int load(pugi::xml_node&);
};


#endif
