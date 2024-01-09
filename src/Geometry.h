#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <map>
#include <string>

#include "pugixml.hpp"
#include "unit.h"
#include "Lists.h"
#include "Region.h"

/// STL triangle structure
#ifdef _WIN32
struct STL_tri {
#else
struct __attribute__((__packed__)) STL_tri {
#endif
    float norm[3];
    float p1[3];
    float p2[3];
    float p3[3];
    short int v;
};

enum draw_mode { MODE_OVERWRITE, MODE_FILL, MODE_CHANGE };

/// Class responsible for constructing the table of flags/NodeTypes
class Geometry {
    const Model * model;
public:
    std::vector<big_flag_t> geom; ///< Main table of flags/NodeType's
    cut_t* Q;
    lbRegion region;       ///< Lattice region
    lbRegion totalregion;  ///< Global Lattice region
    UnitEnv units;         ///< Units object for unit calculations
    Geometry(const lbRegion& r, const lbRegion& tr, const UnitEnv& units_, const Model * model_);
    ~Geometry();
    int load(pugi::xml_node& geometry_node, const std::map<std::string, int>& zone_map);
    void writeVTI(const char* filename);

private:
    big_flag_t fg;              ///< Foreground flag used for filling
    big_flag_t fg_mask;         ///< Foreground flag mask used for filling
    draw_mode fg_mode;      ///< Foreground flag drawing mode
    pugi::xml_node fg_xml;  ///< Foreground flag XML element
    int setFlag(const pugi::char_t* name);
    int setMask(const pugi::char_t* name);
    int setMode(const pugi::char_t* mode);
    int Draw(pugi::xml_node&);
    void setZone(int zone_number);
    int loadZone(const char* name);
    int loadSTL(lbRegion reg, pugi::xml_node n);
    int loadSweep(lbRegion reg, pugi::xml_node n);
    int transformSTL(int, STL_tri*, pugi::xml_node n);
    lbRegion getRegion(const pugi::xml_node& node);
    int val(pugi::xml_attribute attr, int def);
    double val(pugi::xml_attribute attr, double def);
    int val(pugi::xml_attribute attr);
    int val_p(pugi::xml_attribute attr, char* prefix);
    double val_d(pugi::xml_attribute attr);
    big_flag_t Dot(int x, int y, int z);
    void ActivateCuts();
};

#endif
