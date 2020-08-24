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
}

int Connectivity::load(pugi::xml_node & node) {
    int ret;
    
    output("Loading connectivity information ...\n");
    if(!node.attribute("file")) {
        error("No 'file' attribute in ArbitraryLattice element in xml conf\n");
    }

    FILE* cxnFile = fopen(node.attribute("file").value(), "rb");
    if(cxnFile == NULL) {
        error("Connection file can't be opened\n");
        return -1;
    }
    char buffer[20];
    // read header information
    ret = fscanf(cxnFile, "LATTICESIZE %d\n", &latticeSize);
    ret = fscanf(cxnFile, "BASE_LATTICE_DIM %d %d %d\n", &x, &y, &z);
    ret = fscanf(cxnFile, "d %d\n", &d);
    ret = fscanf(cxnFile, "Q %d\n", &Q);
    ret = fscanf(cxnFile, "MASK %s\n", buffer);
    ret = fscanf(cxnFile, "NODES\n");

    ModelBase::NodeTypeFlags::const_iterator it = model->nodetypeflags.ByName(buffer);
    if (it == model->nodetypeflags.end()) {
        ERROR("Unknown flag (in xml): %s\n", buffer);
        return -1;
    }
    big_flag_t defMask = it->flag;

    // allocate memory for connectivity / nodetype matrices
    connectivity = (size_t*) malloc(latticeSize * Q * sizeof(size_t));
    geom = (big_flag_t*) malloc(latticeSize * sizeof(big_flag_t));
    coords = (vector_t*) malloc(latticeSize * sizeof(vector_t));
    
    // read in node connectivity data from file
    for(size_t i = 0; i < latticeSize; i++) {
        char nodeType[20];
        float x, y, z;
        size_t nid;

        // first scan for the nid, nodetype, coords
        ret = fscanf(cxnFile, "%zu %s %e %e %e ", &nid, nodeType, &x, &y, &z);
        // can't fscan a float into a real_t so use intermediate vars
        vector_t w;
        w.x = x;
        w.y = y;
        w.z = z;
        coords[i] = w;
        //coords[3*i] = x;
        //coords[3*i + 1] = y;
        //coords[3*i + 2] = z;

        // find the flag corresponding to our nodeType
        it = model->nodetypeflags.ByName(nodeType);
        if (it == model->nodetypeflags.end()) {
            ERROR("Unknown flag (in xml): %s\n", nodeType);
            return -1;
        }
        // set it to that in the nodetype array
        // if it's a wall, don't set it to mrt as well
        // if it's anything else, & it with mrt.. will need to think of a more elegant way to handle this when I generalise the pre-processor
        if(it->flag == (0x000a))
            geom[i] = it->flag;
        else
            geom[i] = it->flag | defMask;

        // next scan in the connectivity - have to scan an unknown number of integers
        for(int q = 0; q < Q - 1; q++) {
            ret = fscanf(cxnFile, "%zu ", &connectivity[(q * latticeSize) + i]);
        }
        // do the last scan separate to get the newline
        ret = fscanf(cxnFile, "%zu\n", &connectivity[((Q - 1) * latticeSize) + i]);
    }

    // TODO: need to load flags yet, will likely have to do some sort of string->flag_t dict, or change the nodeType field to a number
    fclose(cxnFile);
}