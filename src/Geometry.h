
class Geometry {
public:
  flag_t * geom;
  lbRegion region;
  Geometry(const lbRegion& r);
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
  flag_t Dot(int x, int y, int z);
};
