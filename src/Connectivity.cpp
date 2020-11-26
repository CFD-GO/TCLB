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
#include "utils.h"

#define HANDLE_IOERR(x,n) if ((x) != n) { error("Error in fscanf (2).\n"); return -1; }


Connectivity::Connectivity(const lbRegion & r, const lbRegion & tr, const UnitEnv &units_, ModelBase * model_):model(model_), region(r), totalregion(tr), units(units_)
{  
    // memory allocation is done in the load() function where we know the size of the lattice
    output("Initialising connectivity");
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
        // give this group
        pugi::xml_attribute group = z.attribute("group");
        if(group) {
            if(GroupsToNodeTypes.count(group.value()) > 0)
                GroupsToNodeTypes[group.value()] |= it->flag;
            else
                GroupsToNodeTypes[group.value()] = it->flag;
        }
        
    }

    if(!node.attribute("file")) {
        error("No 'file' attribute in ArbitraryLattice element in xml conf\n");
    }
    cellDataOutput = false;
    if(node.attribute("cellData")) {
        cellDataOutput = true;
        printf("found the cell data attr\n");
    }

    FILE* cxnFile = fopen_gz(node.attribute("file").value(), "rb");
    if(cxnFile == NULL) {
        error("Connection file can't be opened\n");
        return -1;
    }
    char buffer[20];
    DEBUG_M;
    // read header information
    HANDLE_IOERR(fscanf(cxnFile, "LATTICESIZE %zu\n", &latticeSize),1);
    HANDLE_IOERR(fscanf(cxnFile, "BASE_LATTICE_DIM %d %d %d\n", &nx, &ny, &nz),3);
    HANDLE_IOERR(fscanf(cxnFile, "d %d\n", &d),1);
    HANDLE_IOERR(fscanf(cxnFile, "Q %d\n", &Q),1);
    HANDLE_IOERR(fscanf(cxnFile, "OFFSET_DIRECTIONS\n"),0);
    DEBUG_M;

    // allocate the table of offsets
    connectivityDirections = (int*) malloc(Q * 3 * sizeof(int));
    // read the order of offsets
    for(int q = 0; q < Q; q++) {
        HANDLE_IOERR(fscanf(cxnFile, "[%d,%d,%d]", &connectivityDirections[3*q], &connectivityDirections[3*q + 1], &connectivityDirections[3*q + 2]),3);
        if(q < Q-1) {
            HANDLE_IOERR(fscanf(cxnFile, ","),0); // move the file pointer past the ,
        } else {
            HANDLE_IOERR(fscanf(cxnFile, "\n"),0);
        }
    }
    DEBUG_M;
    // initialise the max/min variables -- note: assumes we never stream further away than -1 -> +1.. should be -MAX_INT, +MAX_INT to be perfectly correct
    mindx = 0;
    mindy = 0;
    mindz = 0;
    maxdx = 0;
    maxdy = 0;
    maxdz = 0;
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
    for(int q = 0; q < ndx * ndy * ndz; q++) {
        if (tmp[q] < 0) {
            error("Some directions not filled by the connectivity file\n");
            return -1;
        }
    }
    // get rid of old connectivity array and set to new tmp matrix form
    free(connectivityDirections);
    connectivityDirections = tmp;
    DEBUG_M;

    //HANDLE_IOERR(fscanf(cxnFile, "MASK %s\n", buffer));
    HANDLE_IOERR(fscanf(cxnFile, "NODES\n"),0);

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
        HANDLE_IOERR(fscanf(cxnFile, "%zu %e %e %e ", &nid, &x, &y, &z),4);
        // can't fscan a float into a real_t so use intermediate vars
        vector_t w;
        w.x = x;
        w.y = y;
        w.z = z;
        coords[i] = w;
        geom[i] = 0;

        // next scan in the connectivity - have to scan an unknown number of integers
        for(int q = 0; q < Q; q++) {
            HANDLE_IOERR(fscanf(cxnFile, "%zu ", &connectivity[(q * latticeSize) + i]),1);
        }

        // now read the labels on each node
        int nlabels;
        char label[200];
        // read in the number of labels we have
        HANDLE_IOERR(fscanf(cxnFile, "%d", &nlabels),1);
        // read in our labels after that
        for(int j = 0; j < nlabels; j++) {
            HANDLE_IOERR(fscanf(cxnFile, " %s", label),1);
            // see if we have this label in our mapping
            if(GroupsToNodeTypes.count(label) > 0) {
                // if we do, |= that onto our current NodeType value
                geom[i] |= GroupsToNodeTypes[label];
            } else {
                ERROR("Unknown group label (in connectivity file): %s\n", label);
                return -1;
            }
        }

        HANDLE_IOERR(fscanf(cxnFile, "\n"),0);
    }
    DEBUG_M;

    fclose(cxnFile);

    if(cellDataOutput) {

        FILE* cellFile = fopen_gz(node.attribute("cellData").value(), "rb");

        HANDLE_IOERR(fscanf(cellFile, "N_POINTS %zu\n", &nPoints),1);
        HANDLE_IOERR(fscanf(cellFile, "N_CELLS %zu\n", &nCells),1);

        pointData = (real_t*) malloc(nPoints *3* sizeof(real_t));
        cellConnectivity = (size_t*) malloc(nCells * 8 * sizeof(size_t));
        cellOffsets = (size_t*) malloc(nCells * sizeof(size_t));
        cellTypes = (unsigned char*) malloc(nCells * sizeof(unsigned char));
        HANDLE_IOERR(fscanf(cellFile, "POINTS\n"),0);

        for(size_t i = 0; i < nPoints; i++) {
            float x, y, z;
            HANDLE_IOERR(fscanf(cellFile, "%e %e %e\n", &x, &y, &z),3);
            pointData[3*i] = x;
            pointData[3*i + 1] = y;
            pointData[3*i + 2] = z;
        }
        
        HANDLE_IOERR(fscanf(cellFile, "CELLS\n"),0);
        for(size_t i = 0; i < nCells; i++) {
            HANDLE_IOERR(fscanf(cellFile, "%zu %zu %zu %zu %zu %zu %zu %zu\n", &cellConnectivity[8*i], &cellConnectivity[8*i + 1], &cellConnectivity[8*i + 2], &cellConnectivity[8*i + 3], &cellConnectivity[8*i + 4], &cellConnectivity[8*i + 5], &cellConnectivity[8*i + 6], &cellConnectivity[8*i + 7]),8);
            cellOffsets[i] = 8*(i+1);
            cellTypes[i] = 12;
        }
        printf("Loaded cell connectivity\n");
    }
    return 0;
}
