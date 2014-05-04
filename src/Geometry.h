#ifndef GEOMETRY_H
#define GEOMETRY_H

#include "unit.h"

/// STL triangle structure
struct __attribute__((__packed__)) STL_tri {
        float norm[3];
        float p1[3];
        float p2[3];
        float p3[3];
        short int v;
};

/// Class responsible for constructing the table of flags/NodeTypes
class Geometry {
public:
  flag_t * geom; ///< Main table of flags/NodeType's
  lbRegion region; ///< Global Lattive region
  UnitEnv units; ///< Units object for unit calculations
  Geometry(const lbRegion& r, const UnitEnv& units_);
  ~Geometry();
  int load(pugi::xml_node&);
  void writeVTI(char * filename);
private:
  flag_t fg; ///< Foreground flag used for filling
  flag_t fg_mask; ///< Foreground flag mask used for filling
  pugi::xml_node fg_xml; ///< Foreground flag XML element
  int setFlag(const pugi::char_t * name);
  int setMask(const pugi::char_t * name);
  int Draw(pugi::xml_node&);
  int loadZone(const char * name);
  int loadSTL( lbRegion reg, pugi::xml_node n);
  int transformSTL( int, STL_tri*, pugi::xml_node n);
  lbRegion getRegion(const pugi::xml_node& node);
  int val(pugi::xml_attribute attr, int def);
  int val(pugi::xml_attribute attr);
  double val_d(pugi::xml_attribute attr);
  flag_t Dot(int x, int y, int z);
};

#endif