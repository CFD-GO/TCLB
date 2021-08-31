#include "toArb.h"

#include <stdio.h>
#include <cstdlib>
#include <vector>
#include <string>
#include <queue>
#include <map>
#include <set>
#include <math.h>
#include <array>

struct point {
    long int x,y,z;
    bool operator<(const point& p) const {
        if (x<p.x) return true;
        if (x>p.x) return false;
        if (y<p.y) return true;
        if (y>p.y) return false;
        if (z<p.z) return true;
        if (z>p.z) return false;
        return false;
    }
};

typedef std::map<point, size_t> id_map_t;

struct element {
    point p;
    bool interior;
    bool vtu_export;
    std::array<size_t, 27> con;
    std::array<size_t, 8> cell;
};

const int pb_len=80;
int pb_now = 0;

void pb_tick(const size_t i,const size_t n) {
    int now = floor(1.0 *pb_len*i/n);
    if (now != pb_now) {
        printf("[");
        int k=0;
        for (; k<now; k++) printf("=");
        for (; k<pb_len; k++) printf(" ");
        printf("]\r");
        fflush(stdout);
        pb_now = now;
    }
    if (i == n) printf("\n");
}

big_flag_t getFlag(Geometry * geom, const point& p) {
    return geom->geom[geom->region.offset(p.x,p.y,p.z)];
}


int toArbitrary(Solver* solver, ModelBase* model) {
    pugi::xml_attribute attr;
    id_map_t id_map;
    id_map_t point_map;
    std::vector<element> lattice;
    std::vector<point> points;

    lbRegion region = solver->info.region;

    bool exportInteriorOnly; // ! write interior, write for normal TCLB, export biggest components only

    unsigned int error;
    char filename[1024];
    size_t count = 0;
    FILE* f = NULL;

    exportInteriorOnly = true;

    big_flag_t bulk_flag, bulk_mask;
    
	const char * remove_bulk = NULL;
    attr = solver->configfile.child("CLBConfig").attribute("remove_bulk");
    if (attr) {
        remove_bulk = attr.value();
    }
    
    if (remove_bulk != NULL) {
        ModelBase::NodeTypeFlags::const_iterator it = model->nodetypeflags.ByName(remove_bulk); // TODO
        if (it == model->nodetypeflags.end()) {
                ERROR("Unknown flag (in xml): %s\n", remove_bulk); // TODO
                return -1;
        }
        bulk_flag = it->flag;
        bulk_mask = it->group_flag;
    } else {
        bulk_flag = -1; //Nothing will match
        bulk_mask = 0;
    }

    printf("Generating interior:\n");
    for (long int z = 0; z<region.nz; z++) {
        for (long int y=0; y<region.ny; y++) {
            for (long int x=0; x<region.nx; x++) {
                point p;
                p.x=x;
                p.y=y;
                p.z=z;
                if ((getFlag(solver->geometry, p) & bulk_mask) != bulk_flag) { //TODO
                    element el;
                    el.interior = true;
                    el.p = p;
                    lattice.push_back(el);
                } else {
                }
                count++;
            }
        }
        pb_tick(z+1,region.nz);
    }
    printf("Interior size: %ld / %ld\n", lattice.size(), count);
    printf("Generating map:\n");
    for (size_t i=0; i<lattice.size(); i++) {
        point p = lattice[i].p;
        if (id_map.find(p) != id_map.end()) {
            fprintf(stderr, "Element in the map\n");
            return -1;
        }
        id_map[p] = i;
        pb_tick(i+1,lattice.size());
    }

    std::queue< size_t > Q;

    printf("Generating connections:\n");
    for (size_t i=0; i<lattice.size(); i++) {
        point p = lattice[i].p;
        int k=0;
        for (int z=-1; z<=1; z++) {
            for (int y=-1; y<=1; y++) {
                for (int x=-1; x<=1; x++) {
                    point np;
                    size_t id;
                    np.x = (p.x + x + region.nx) % region.nx;
                    np.y = (p.y + y + region.ny) % region.ny;
                    np.z = (p.z + z + region.nz) % region.nz;
                    id_map_t::iterator it = id_map.find(np);
                    if (it != id_map.end()) {
                        id = it->second;
                    } else if (lattice[i].interior) {
                        element el;
                        el.p = np;
                        el.interior = false;
                        id = lattice.size();
                        id_map[el.p] = id;
                        lattice.push_back(el);
                    } else {
                        id = i;
                    }
                    lattice[i].con[k] = id;
                    k++;
                }
            }
        }
        lattice[i].vtu_export = (lattice[i].interior || (!exportInteriorOnly));
        k=0;
        if (lattice[i].vtu_export) {
            for (int z=0; z<=1; z++) {
                for (int y=0; y<=1; y++) {
                    for (int x=0; x<=1; x++) {
                        point np;
                        size_t id;
                        np.x = p.x + x;
                        np.y = p.y + y;
                        np.z = p.z + z;
                        id_map_t::iterator it = point_map.find(np);
                        if (it != point_map.end()) {
                            id = it->second;
                        } else {
                            id = points.size();
                            point_map[np] = id;
                            points.push_back(np);
                        }
                        lattice[i].cell[k] = id;
                        k++;
                    }
                }
            }
        }
        pb_tick(i+1,lattice.size());
    }
    printf("Writing connectivity:\n");
    char cxnString[STRING_LEN];
    solver->outGlobalFile("ARB",".cxn",cxnString);
    f = fopen(cxnString,"w");
    fprintf(f,"LATTICESIZE %lu\n",lattice.size());
    fprintf(f,"BASE_LATTICE_DIM %d %d %d\n",20,20,20); // this is a mockup
    fprintf(f,"d 3\n");
    fprintf(f,"Q 27\n");
    fprintf(f,"OFFSET_DIRECTIONS\n");
    for (int z=-1; z<=1; z++) {
        for (int y=-1; y<=1; y++) {
            for (int x=-1; x<=1; x++) {
                fprintf(f,"[%d,%d,%d]",x,y,z);
                if ((x!=1) || (y!=1) || (z!=1))
                    fprintf(f,",");
                else
                    fprintf(f,"\n");
            }
        }
    }
    fprintf(f,"NODES\n");
    double spacing = 1/solver->units.alt("m");
    for (size_t i=0; i<lattice.size(); i++) {
        fprintf(f,"%lu %lg %lg %lg",
            i,
            (0.5+lattice[i].p.x) * spacing,
            (0.5+lattice[i].p.y) * spacing,
            (0.5+lattice[i].p.z) * spacing);
        for (int k=0; k<27; k++) fprintf(f," %lu", lattice[i].con[k]);
        std::vector<std::string> groups;
        big_flag_t flag = getFlag(solver->geometry, lattice[i].p);
        for (ModelBase::NodeTypeFlags::const_iterator it = model->nodetypeflags.begin(); it != model->nodetypeflags.end(); it++) {
            if (it->flag == 0) continue;
            if ((flag & it->group_flag) == it->flag) groups.push_back(it->name);
        }
        int z = (flag & model->settingzones.flag) >> model->settingzones.shift;
        if (z != 0) {
            for (std::map<std::string, int>::const_iterator it = solver->geometry->SettingZones.begin(); it != solver->geometry->SettingZones.end(); it++) {
                if (z == it->second) groups.push_back("Z_" + it->first);
            }
        }
        if (! lattice[i].vtu_export) groups.push_back("HIDE");
        fprintf(f," %d", groups.size());
        for (std::vector<std::string>::const_iterator it = groups.begin(); it != groups.end(); it++) {
            fprintf(f," %s", it->c_str());
        }
        fprintf(f,"\n", groups.size());
        pb_tick(i+1,lattice.size());
    }
    fclose(f);
    printf("Writing points:\n");
    char cellString[STRING_LEN];
    solver->outGlobalFile("ARB",".cell",cellString);
    f = fopen(cellString,"w");
    fprintf(f,"N_POINTS %lu\n",points.size());
    size_t cells = 0;
    for (size_t i=0; i<lattice.size(); i++) if (lattice[i].vtu_export) cells++;

    fprintf(f,"N_CELLS %lu\n",cells);
    fprintf(f,"POINTS\n");
    for (size_t i=0; i<points.size(); i++) {
        fprintf(f,"%lg %lg %lg\n", spacing*points[i].x, spacing*points[i].y, spacing*points[i].z);
        pb_tick(i+1,points.size());
    }
    fprintf(f,"CELLS\n");
    printf("Writing cells:\n");
    size_t k=0;
    for (size_t i=0; i<lattice.size(); i++) if (lattice[i].vtu_export) {
        fprintf(f,"%lu %lu %lu %lu %lu %lu %lu %lu\n",
            lattice[i].cell[0],
            lattice[i].cell[1],
            lattice[i].cell[3],
            lattice[i].cell[2],
            lattice[i].cell[4],
            lattice[i].cell[5],
            lattice[i].cell[7],
            lattice[i].cell[6]);
        pb_tick(k+1,cells);
        k++;
    }
    fclose(f);

    pugi::xml_document restartfile;
	for (pugi::xml_node n = solver->configfile.first_child(); n; n = n.next_sibling()) restartfile.append_copy(n);
    pugi::xml_node n0 = restartfile.child("CLBConfig");
	pugi::xml_node n1 = n0.child("Geometry");
    if (!n1){
        ERROR("No geometry node in xml - this should not happen");
        return -1;
    }
    
    attr = n0.attribute("toArb"); if (attr) n0.remove_attribute(attr);
    attr = n0.attribute("remove_bulk"); if (attr) n0.remove_attribute(attr);
	pugi::xml_node n2 = n0.insert_child_before("ArbitraryLattice", n1);
    n0.remove_child(n1);
	n2.append_attribute("file").set_value(cxnString);
    n2.append_attribute("cellData").set_value(cellString);
    for (ModelBase::NodeTypeFlags::const_iterator it = model->nodetypeflags.begin(); it != model->nodetypeflags.end(); it++) {
        if (it->flag == 0) continue;
        pugi::xml_node n3 = n2.append_child(it->name.c_str());
        n3.append_attribute("group").set_value(it->name.c_str());
    }
    for (std::map<std::string, int>::const_iterator it = solver->geometry->SettingZones.begin(); it != solver->geometry->SettingZones.end(); it++) {
        if (it->second == 0) continue;
        pugi::xml_node n3 = n2.append_child("None");
        n3.append_attribute("group").set_value(("Z_" + it->first).c_str());
        n3.append_attribute("name").set_value(it->first.c_str());
    }
    char rstString[STRING_LEN];
    solver->outGlobalFile("ARB",".xml",rstString);
	restartfile.save_file( rstString );

    return 0;
}
