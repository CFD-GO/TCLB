#ifndef CONNECTIVITY_H
#define CONNECTIVITY_H

#include "unit.h"
#include "Lists.h"
#include "pugixml.hpp"


class Connectivity {
    ModelBase* model;
public:
    big_flag_t * geom; ///< Main table of flags/NodeType's
    size_t * connectivity; ///< Main connectivity matrix
    size_t latticeSize; ///< Number of nodes in the arbitrary lattice
    int d, Q;
    int nx, ny, nz; ///< Dimensions of the base lattice
    vector_t * coords; ///< Table of coordinates of each node
    bool cellDataOutput;
    size_t nPoints;
    size_t nCells;
    real_t * pointData;
    size_t * cellConnectivity;
    size_t * cellOffsets;
    std::map<std::string,big_flag_t> GroupsToNodeTypes;
    std::map<std::string,std::string> GroupsToZones;
    std::map<std::string,int> SettingZones;
    int * connectivityDirections;
    int ndx, ndy, ndz, mindx, mindy, mindz, maxdx, maxdy, maxdz;
    unsigned char * cellTypes;
    lbRegion region; ///< Local lattice region
    lbRegion totalregion; ///< Global lattice region
    UnitEnv units; ///< Units object for unit calculations

    Connectivity(const lbRegion& r, const lbRegion& tr, const UnitEnv& units_, ModelBase * model_);
    int setZone(const pugi::char_t * name);
    int load(pugi::xml_node&);
};


#endif
