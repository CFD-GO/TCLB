#include <stdlib.h>
#include <cstring>
#include <iostream>
#include "pugixml.hpp"

#include "cross.h"
#include "types.h"
#include "Region.h"
#include "Connectivity.h"
#include "Global.h"
#include "def.h"

Connectivity::Connectivity(const lbRegion & r, const lbRegion & tr, const UnitEnv &units_, ModelBase * model_):model(model_), region(r), totalregion(tr), units(units_)
{  
    // memory allocation is done in the load() function where we know the size of the lattice
    output("Initialising connectivity");
    SettingZones["DefaultZone"] = 0;
}


/**
 * Add new zone to our zone list ZoneSettings
 **/
int Connectivity::setZone(const pugi::char_t * name) {
    int ZoneNumber;
    if (SettingZones.count(name) > 0) {
        ZoneNumber = SettingZones[name];
    } else {
        ZoneNumber = SettingZones.size();
        debug1("Setting new zone: %s -> %d\n", name, ZoneNumber);
        SettingZones[name] = ZoneNumber;
    }
    // not sure about if we need this part - will investigate more
    //assert(ZoneNumber < model->settingzones.capacity);
    //fg      = (fg      &(~ model->settingzones.flag )) |  (ZoneNumber << model->settingzones.shift);
    //fg_mask =  fg_mask |   model->settingzones.flag;

    return 0;
}


int Connectivity::load(pugi::xml_node & node) {
    int ret;
    
    output("Loading connectivity information ...\n");
    
    // first read mappings from connectivity file 'labels' to nodetypes from child nodes
    for(pugi::xml_node z = node.first_child(); z; z = z.next_sibling()) {
        // find the nodetype flag for the child name in xml config
        ModelBase::NodeTypeFlags::const_iterator it;
        it = model->nodetypeflags.ByName(z.name());
        if (it == model->nodetypeflags.end()) {
            ERROR("Unknown flag (in xml): %s\n", z.name());
            return -1;
        }
        //printf("Flag found for %s: %d\n", z.name(), it->flag);
        // get the group tag of this node
        pugi::xml_attribute group = z.attribute("group");
        if(group) {
            if(GroupsToNodeTypes.count(group.value()) > 0)
                GroupsToNodeTypes[group.value()] |= it->flag;
            else
                GroupsToNodeTypes[group.value()] = it->flag;
        }
        // see if this node also has a zone tag
        pugi::xml_attribute zone = z.attribute("name");
        if(zone && group) {
            // if it does, link the group to the zone - I think we assume it's one-to-one at the moment but will probably have to change this
            GroupsToZones[group.value()] = zone.value();
            // call function to add to SettingZones table
            setZone(zone.value());
        }
        
    }

    // iterate over map and read mappings
    std::map<std::string, std::string>::iterator gzit;
    for(gzit = GroupsToZones.begin(); gzit != GroupsToZones.end(); gzit++) {
        //printf("%s: %s\n", gzit->first, gzit->second);
        std::cout << gzit->first << ": " << gzit->second << "\n";
    }

    if(!node.attribute("file")) {
        error("No 'file' attribute in ArbitraryLattice element in xml conf\n");
    }
    cellDataOutput = false;
    if(node.attribute("cellData")) {
        cellDataOutput = true;
        printf("found the cell data attr\n");
    }

    FILE* cxnFile = fopen(node.attribute("file").value(), "rb");
    if(cxnFile == NULL) {
        error("Connection file can't be opened\n");
        return -1;
    }
    char buffer[20];

    // read header information
    ret = fscanf(cxnFile, "LATTICESIZE %d\n", &latticeSize);
    ret = fscanf(cxnFile, "BASE_LATTICE_DIM %d %d %d\n", &nx, &ny, &nz);
    ret = fscanf(cxnFile, "d %d\n", &d);
    ret = fscanf(cxnFile, "Q %d\n", &Q);
    ret = fscanf(cxnFile, "OFFSET_DIRECTIONS\n");

    // allocate the table of offsets
    connectivityDirections = (int*) malloc(Q * 3 * sizeof(int));
    // read the order of offsets
    for(int q = 0; q < Q; q++) {
        ret = fscanf(cxnFile, "[%d,%d,%d]", &connectivityDirections[3*q], &connectivityDirections[3*q + 1], &connectivityDirections[3*q + 2]);
        if(q < Q-1)
            fscanf(cxnFile, ","); // move the file pointer past the ,
        else
            fscanf(cxnFile, "\n");
    }
    // initialise the max/min variables -- note: assumes we never stream further away than -1 -> +1.. should be -MAX_INT, +MAX_INT to be perfectly correct
    mindx = 1;
    mindy = 1;
    mindz = 1;
    maxdx = -1;
    maxdy = -1;
    maxdz = -1;
    // determine max/min connectivity directions
    for(int q = 0; q < Q; q++) {
        if(connectivityDirections[3*q] < mindx)
            mindx = connectivityDirections[3*q];
        if(connectivityDirections[3*q + 1] < mindy)
            mindy = connectivityDirections[3*q + 1];
        if(connectivityDirections[3*q + 2] < mindz)
            mindz = connectivityDirections[3*q + 2];

        if(connectivityDirections[3*q] > maxdx)
            maxdx = connectivityDirections[3*q];
        if(connectivityDirections[3*q + 1] > maxdy)
            maxdy = connectivityDirections[3*q + 1];
        if(connectivityDirections[3*q + 2] > maxdz)
            maxdz = connectivityDirections[3*q + 2];
    }
    ndx = maxdx - mindx + 1;
    ndy = maxdy - mindy + 1;
    ndz = maxdz - mindz + 1;
    // reallocate and reset connectivity directions to be a ndx * ndy * ndz matrix
    int* tmp = (int*) malloc(ndx * ndy * ndz * sizeof(int));
    // set it all to -1 so we can identify offsets we don't know
    memset(tmp, -1, ndx * ndy * ndz * sizeof(int));
    
    // load values in connectivityDirections in here
    for(int q = 0; q < Q; q++) {
        tmp[(connectivityDirections[3*q] - mindx) + ((connectivityDirections[3*q + 1] - mindy) * ndx) + ((connectivityDirections[3*q + 2] - mindz) * ndx * ndy)] = q;
    }
    // get rid of old connectivity array and set to new tmp matrix form
    free(connectivityDirections);
    connectivityDirections = tmp;

    //ret = fscanf(cxnFile, "MASK %s\n", buffer);
    ret = fscanf(cxnFile, "NODES\n");

    // allocate memory for connectivity / nodetype matrices
    connectivity = (size_t*) malloc(latticeSize * Q * sizeof(size_t));
    geom = (big_flag_t*) malloc(latticeSize * sizeof(big_flag_t));
    coords = (vector_t*) malloc(latticeSize * sizeof(vector_t));
    memset(geom, 0, latticeSize * sizeof(big_flag_t));
    
    // read in node connectivity data from file
    for(size_t i = 0; i < latticeSize; i++) {
        char nodeType[20];
        float x, y, z;
        size_t nid;

        // first scan for the nid, nodetype, coords
        ret = fscanf(cxnFile, "%zu %e %e %e ", &nid, &x, &y, &z);
        // can't fscan a float into a real_t so use intermediate vars
        vector_t w;
        w.x = x;
        w.y = y;
        w.z = z;
        coords[i] = w;

        // next scan in the connectivity - have to scan an unknown number of integers
        for(int q = 0; q < Q; q++) {
            ret = fscanf(cxnFile, "%zu ", &connectivity[(q * latticeSize) + i]);
            //ret = fscanf(cxnFile, "%zu ", &connectivity[q + (i * Q)]); // TEMP - changed to AoS
        }

        // now read the labels on each node
        int nlabels;
        char label[20];
        // read in the number of labels we have
        ret = fscanf(cxnFile, "%d", &nlabels);
        // read in our labels after that
        for(int j = 0; j < nlabels; j++) {
            ret = fscanf(cxnFile, " %s", &label);
            // see if we have this label in our mapping
            if(GroupsToNodeTypes.count(label) > 0) {
                // if we do, |= that onto our current NodeType value
                geom[i] |= GroupsToNodeTypes[label];
                
            } else {
                ERROR("Unknown group label (in connectivity file): %s\n", label);
                return -1;
            }
            // see if we have this label in our mapping from groups to zones
            if(GroupsToZones.count(label) > 0) {
                if(SettingZones.count(GroupsToZones[label]) > 0) {
                    int zoneNum = SettingZones[GroupsToZones[label]];
                    geom[i] |= zoneNum << model->settingzones.shift;
                }
                // fg      = (fg      &(~ model->settingzones.flag )) |  (ZoneNumber << model->settingzones.shift); need to do something like this
            }
        }

        if(x == -1 && y == 10 && z == 10) {
            printf("At %e, %e, %e, geom[i] is: %d\n", x, y, z, geom[i]);
        }
        if(x == 0 && y == -1 && z == 20) {
            printf("At %e, %e, %e, geom[i] is: %d\n", x, y, z, geom[i]);
        }

        ret = fscanf(cxnFile, "\n");
    }

    fclose(cxnFile);

    if(cellDataOutput) {

        FILE* cellFile = fopen(node.attribute("cellData").value(), "rb");

        ret = fscanf(cellFile, "N_POINTS %zu\n", &nPoints);
        ret = fscanf(cellFile, "N_CELLS %zu\n", &nCells);

        pointData = (real_t*) malloc(nPoints *3* sizeof(real_t));
        cellConnectivity = (size_t*) malloc(nCells * 8 * sizeof(size_t));
        cellOffsets = (size_t*) malloc(nCells * sizeof(size_t));
        cellTypes = (unsigned char*) malloc(nCells * sizeof(unsigned char));
        ret = fscanf(cellFile, "POINTS\n");

        for(int i = 0; i < nPoints; i++) {
            float x, y, z;
            ret = fscanf(cellFile, "%e %e %e\n", &x, &y, &z);
            pointData[3*i] = x;
            pointData[3*i + 1] = y;
            pointData[3*i + 2] = z;
        }
        
        ret = fscanf(cellFile, "CELLS\n");
        for(int i = 0; i < nCells; i++) {
            ret = fscanf(cellFile, "%zu %zu %zu %zu %zu %zu %zu %zu\n", &cellConnectivity[8*i], &cellConnectivity[8*i + 1], &cellConnectivity[8*i + 2], &cellConnectivity[8*i + 3], &cellConnectivity[8*i + 4], 
                                                                &cellConnectivity[8*i + 5], &cellConnectivity[8*i + 6], &cellConnectivity[8*i + 7]);
            cellOffsets[i] = 8*(i+1);
            cellTypes[i] = 12;
        }
        printf("Loaded cell connectivity\n");
    }
}