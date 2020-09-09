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
    ret = fscanf(cxnFile, "BASE_LATTICE_DIM %d %d %d\n", &x, &y, &z);
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

    ret = fscanf(cxnFile, "MASK %s\n", buffer);
    ret = fscanf(cxnFile, "NODES\n");

    big_flag_t defMask = 0;
    ModelBase::NodeTypeFlags::const_iterator it;
    if(strcmp(buffer, "NONE") != 0) {
        it = model->nodetypeflags.ByName(buffer);
        if (it == model->nodetypeflags.end()) {
            ERROR("Unknown flag (in xml): %s\n", buffer);
            return -1;
        }
        defMask = it->flag;
    }

    

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

        if(strcmp(nodeType, "NONE") != 0) {
            // find the flag corresponding to our nodeType
            it = model->nodetypeflags.ByName(nodeType);
            if (it == model->nodetypeflags.end()) {
                ERROR("Unknown flag (in xml): %s\n", nodeType, nid);
                return -1;
            }
            // set it to that in the nodetype array
            // if it's a wall, don't set it to mrt as well
            // if it's anything else, & it with mrt.. will need to think of a more elegant way to handle this when I generalise the pre-processor
            if(strcmp(nodeType, "Wall") == 0)
                geom[i] = it->flag;
            else
                geom[i] = it->flag | defMask;

        }
        
        

        // next scan in the connectivity - have to scan an unknown number of integers
        for(int q = 0; q < Q - 1; q++) {
            ret = fscanf(cxnFile, "%zu ", &connectivity[(q * latticeSize) + i]);
        }
        // do the last scan separate to get the newline
        ret = fscanf(cxnFile, "%zu\n", &connectivity[((Q - 1) * latticeSize) + i]);
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
