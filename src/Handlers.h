#include "pugixml.hpp"

class Solver;

class lbCallback {
	public:
	int startIter;
	int everyIter;
	pugi::xml_node node;
	virtual int DoIt(Solver*);
};

lbCallback * getHandler(pugi::xml_node);


