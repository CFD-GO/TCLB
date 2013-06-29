#include "pugixml.hpp"

#define HANDLER_CALLBACK 0x01
#define HANDLER_ACTION 0x02
#define HANDLER_GENERIC 0x04
#define HANDLER_CONTAINER 0x08

class Solver;

class vHandler {
	public:
	int startIter;
	int everyIter;
	pugi::xml_node node;
	virtual int Init(Solver*);
	virtual int DoIt(Solver*);
	virtual int Type();
};

vHandler * getHandler(pugi::xml_node);

class Handler {
private:
public:
	vHandler * hand;
	int *ref;
//public:
	inline Handler(pugi::xml_node node) {
		hand = getHandler(node);
		ref = new int;
		*ref=1;
		DEBUG0(printf("H: create\n");)
	}
	inline Handler(const Handler & that) {
		hand = that.hand;
		ref = that.ref;
		(*ref)++;
		DEBUG0(printf("H: + %d\n", *ref);)
	}
	inline const int Init(Solver* solver) { return hand->Init(solver); }
	inline const int DoIt(Solver* solver) { return hand->DoIt(solver); }
	inline const int Type() { return hand->Type(); }
	inline const bool Now(int iter) {
		if (hand->everyIter) {
			iter -= hand->startIter;
			return floor((iter)/hand->everyIter) > floor((iter-1)/hand->everyIter);
		} else return false;
	}
	inline const int Next(int iter) {
		if (hand->everyIter) {
			iter -= hand->startIter;
			int k = floor((iter)/hand->everyIter);
			return (k+1) * hand->everyIter - iter;
		} else return -1;
	}
	inline const int Prev(int iter) {
		if (hand->everyIter) {
			iter -= hand->startIter;
			int k = floor((iter-1)/hand->everyIter);
			return iter - (k-1) * hand->everyIter;
		} else return -1;
	}
	inline Handler & operator=(const Handler & that) {
		hand = that.hand;
		ref = that.ref;
		(*ref)++;
		DEBUG0(printf("H: + %d\n", *ref);)
		return *this;
	}
	inline ~Handler() {
		(*ref)--;
		DEBUG0(printf("H: - %d\n", *ref);)
		if (ref <= 0) {
			delete hand;
			delete ref;
		}
	}
	inline operator bool () { return hand != NULL; }
};



