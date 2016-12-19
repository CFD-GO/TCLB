#ifndef Handler_H
#define Handler_H

#include "Handlers/vHandler.h"
#include "Handlers/NullHandler.h"
#include "HandlerFactory.h"

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
		hand = HandlerFactory::Produce(node);
		if (hand == NULL) {
			hand = new NullHandler();
		}
		hand->solver = solver_;
		hand->node = node;
		int ret = hand->Init();
		if (ret) {
			delete hand;
			hand = NULL;
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



#endif // Handler_H