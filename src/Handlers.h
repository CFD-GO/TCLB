#include "Consts.h"
#include "pugixml.hpp"

#define HANDLER_CALLBACK  0x01
#define HANDLER_ACTION    0x02
#define HANDLER_DESIGN    0x03
#define HANDLER_GENERIC   0x10
#define HANDLER_CONTAINER 0x20

class Solver;

/// Main Handler prototype (virtual) class
/**
	All the handlers inherit from this class.
	Handler defines the behavior of the code
	for specific elements of the XML configuration file.
*/
class vHandler {
	public:
	int startIter; ///< Iteration at which the Handler was initialized
	double everyIter; ///< Interval by which it should be activated
	pugi::xml_node node; ///< XML element connected to the Handler
	Solver* solver; ///< The solver object connected to the Handler
	virtual int Init(); ///< Initialize the Handler
	virtual int DoIt(); ///< Do what have to be done
	virtual int Finish(); ///< Finalize the Handler
	virtual int Type(); ///< Return the type of the Handler
	virtual int NumberOfParameters(); ///< Return the type of the Handler
	virtual int GetParameters(double *); ///< Return the type of the Handler
	virtual int SetParameters(const double *); ///< Return the type of the Handler
	virtual int GetGradient(double *); ///< Return the type of the Handler
/// Check if Now is the time to run this Handler
/**
	Checks if now is the time to DoIt for this Handler
	\param iter Iteration number for which to check
	\return True if the iteration is right to DoIt
*/
	inline const bool Now(int iter) {
		if (everyIter) {
			iter -= startIter;
			return floor((iter)/everyIter) > floor((iter-1)/everyIter);
		} else return false;
	}
/// Check what will be the next iteration to run this Handler
/**
	Calculate the next iteration to DoIt
	\param iter Iteration from which to calculate the next one
	\return Number of the next iteration to DoIt
*/
	inline const int Next(int iter) {
		if (everyIter) {
			iter -= startIter;
			int k = floor((iter)/everyIter);
			return - floor(-(k+1) * everyIter) - iter;
		} else return -1;
	}
	
/// Check what will be the next iteration to run this Handler in reverse
/**
	Calculate the next iteration to DoIt when running backwards
	\param iter Iteration from which to calculate the next one
	\return Number of the next iteration to DoIt
*/
	inline const int Prev(int iter) {
		if (everyIter) {
			iter -= startIter;
			int k = floor((iter-1)/everyIter);
			return iter + floor(-k * everyIter);
		} else return -1;
	}
};

/// Generate a Handler based on a XML element
vHandler * getHandler(pugi::xml_node);

/// Encapsulating Handler class.
/**
	It's kind of a shared pointer for Handlers
*/
class Handler {
private:
public:
	vHandler * hand; ///< Handler
	int *ref; ///< Number of references of the shared pointer
/// Constructs a Handler based on a XML element
	inline Handler(pugi::xml_node node, Solver * solver_) {
		hand = getHandler(node);
		hand->solver = solver_;
		if (hand) {
			int ret = hand->Init();
			if (ret) {
				delete hand;
				hand = NULL;
			}
		}
		ref = new int;
		*ref=1;
		debug0("Handler shared pointer: create\n");
	}
/// Makes another reference of the shared pointer
	inline Handler(const Handler & that) {
		hand = that.hand;
		ref = that.ref;
		(*ref)++;
		debug0("Handler shared pointer++: %d\n", *ref);
	}
/// Dispatches Init on the vHandler
	inline const int Init() { return hand->Init(); }
/// Dispatches DoIt on the vHandler
	inline const int DoIt() { return hand->DoIt(); }
/// Dispatches Init on the vHandler
	inline const int Type() { return hand->Type(); }
/// Dispatches Init on the vHandler
	inline const bool Now(int iter) { return hand->Now(iter); }
/// Dispatches Init on the vHandler
	inline const int Next(int iter) { return hand->Next(iter); }
/// Dispatches Init on the vHandler
	inline const int Prev(int iter) { return hand->Prev(iter); }
/// Makes another reference of the shared pointer
	inline Handler & operator=(const Handler & that) {
		hand = that.hand;
		ref = that.ref;
		(*ref)++;
		debug0("Handler shared pointer++: %d\n", *ref);
		return *this;
	}
/// Gets the vHandler
	vHandler& operator* () { return *hand; }
	vHandler* operator-> () { return hand; }
/// Deletes a shared pointer reference
	inline ~Handler() {
		(*ref)--;
		debug0("Handler shared pointer--: %d\n", *ref);
		if (ref <= 0) {
			if (hand) {
				hand->Finish();
				delete hand;
			}
			delete ref;
		}
	}
/// Checks in the handler is non-null
	inline operator bool () { return hand != NULL; }
};



