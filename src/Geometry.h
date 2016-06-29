#ifndef GEOMETRY_H
#define GEOMETRY_H

#include "unit.h"
#include <map>
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

enum draw_mode {
  MODE_OVERWRITE,
  MODE_FILL,
  MODE_CHANGE
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
  std::map<std::string,int> SettingZones;
private:
  flag_t fg; ///< Foreground flag used for filling
  flag_t fg_mask; ///< Foreground flag mask used for filling
  draw_mode fg_mode; ///< Foreground flag drawing mode
  pugi::xml_node fg_xml; ///< Foreground flag XML element
  int setFlag(const pugi::char_t * name);
  int setMask(const pugi::char_t * name);
  int setMode(const pugi::char_t * mode);
  int setZone(const pugi::char_t * name);
  int Draw(pugi::xml_node&);
  int loadZone(const char * name);
  int loadSTL( lbRegion reg, pugi::xml_node n);
  int loadSweep( lbRegion reg, pugi::xml_node n);
  int transformSTL( int, STL_tri*, pugi::xml_node n);
  lbRegion getRegion(const pugi::xml_node& node);
  int val(pugi::xml_attribute attr, int def);
  double val(pugi::xml_attribute attr, double def);
  int val(pugi::xml_attribute attr);
  int val_p(pugi::xml_attribute attr, char* prefix);
  double val_d(pugi::xml_attribute attr);
  flag_t Dot(int x, int y, int z);
};

#endif
