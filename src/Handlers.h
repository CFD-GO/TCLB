#include "pugixml.hpp"

class Solver;

class Handler {
	public:
	int startIter;
	int everyIter;
	pugi::xml_node node;
	inline Handler(pugi::xml_node node_) : node(node_) {};
	virtual int DoIt(Solver*);
};

Handler * getHandler(pugi::xml_node);


