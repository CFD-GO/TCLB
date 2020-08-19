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
    /*latticeSize = latticeSize;
    geom = new big_flag_t[latticeSize];
    for(size_t i = 0; i < latticeSize; i++) {
        geom[i] = 0;
    }
    output("Creating geom size:%ld\n", latticeSize);*/
    // Will be done in the load() function as we need to read the connectivity file
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
    char buffer[80];
    int d, Q;
    // read header information
    ret = fscanf(cxnFile, "LATTICESIZE %d\n", &latticeSize);
    ret = fscanf(cxnFile, "DIMENSION %d\n", &d);
    ret = fscanf(cxnFile, "Q %d\n", &Q);
    ret = fscanf(cxnFile, "NODES\n");

    // allocate memory for connectivity / nodetype matrices
    connectivity = (size_t*) malloc(latticeSize * Q * sizeof(size_t));
    geom = (big_flag_t*) malloc(latticeSize * sizeof(big_flag_t));

    // read in node connectivity data from file
    

    output("Reading lattice size of %d dim of %d\n", latticeSize, d);
    fclose(cxnFile);
}