#ifndef Handler_H
#define Handler_H

#include "Global.h"
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
		ref = new int;
		*ref=1;
		hand = HandlerFactory::Produce(node);
		if (hand == NULL) {
			ERROR("Unknown Handler: %s",node.name());
			hand = new NullHandler();
		}
		hand->solver = solver_;
		hand->node = node;
		int ret = hand->Init();
		if (ret) {
			delete hand;
			hand = NULL;
		}
	}
/// Makes another reference of the shared pointer
	inline void Handler_inc(const Handler & that) {
		hand = that.hand;
		ref = that.ref;
		(*ref)++;
	}
	inline void Handler_dec() {
		(*ref)--;
		if ((*ref) < 0) {
			ERROR("Handler ptr ref below zero (%p).", hand);
			exit(-1);
		} else if ((*ref) == 0) {
			if (hand) {
				hand->Finish();
				delete hand;
			}
			delete ref;
		}
		hand = NULL;
		ref = NULL;
	}
	inline Handler(const Handler & that) {
		Handler_inc(that);
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
		Handler_dec();
		Handler_inc(that);
		return *this;
	}
/// Gets the vHandler
	vHandler& operator* () { return *hand; }
	vHandler* operator-> () { return hand; }
/// Deletes a shared pointer reference
	inline ~Handler() {
		Handler_dec();
	}
/// Checks in the handler is non-null
	inline operator bool () { return hand != NULL; }
};



#endif // Handler_H