#ifndef GEOMETRY_H
#define GEOMETRY_H

#include "unit.h"

class Geometry {
public:
  flag_t * geom;
  lbRegion region;
  UnitEnv units;
  Geometry(const lbRegion& r, const UnitEnv& units_);
  ~Geometry();
  int load(pugi::xml_node&);
  void writeVTI(char * filename);
private:
  flag_t fg;
  flag_t fg_mask;
  pugi::xml_node fg_xml;
  int setFlag(const pugi::char_t * name);
  int setMask(const pugi::char_t * name);
  int Draw(pugi::xml_node&);
  int loadZone(const char * name);
  lbRegion getRegion(const pugi::xml_node& node);
  int val(pugi::xml_attribute attr, int def);
  int val(pugi::xml_attribute attr);
  flag_t Dot(int x, int y, int z);
};

#endif