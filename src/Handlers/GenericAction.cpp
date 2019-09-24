#include "GenericAction.h"

int GenericAction::Init () {
		stack=0;
//		parSize= -1;
		return Action::Init();
	}


int GenericAction::ExecuteInternal () {
		stack=0;
                for (pugi::xml_node par = node.first_child(); par; par = par.next_sibling()) {
			Handler hand(par, solver);
			if (hand) {
				if (hand.Type() & HANDLER_DESIGN) {
					output("Adding %s to the solver hands\n",(*hand).node.name());
					solver->hands.push_back(hand);
					stack++;
				} else if (hand.Type() & HANDLER_CALLBACK) {
					if ((hand.hand->everyIter != 0) || (hand.Type() & HANDLER_DESIGN)) {
						debug1("adding %lf\n", hand.hand->everyIter);
						output("Adding %s to the solver hands\n",(*hand).node.name());
						solver->hands.push_back(hand);
						stack++;
					} else {
						if ( 0 != hand.DoIt() ) {
                            error("Handler call error: %s", par.name());
                            return -1;
                        };
					}
				}
			} else {
				ERROR("Something wrong in %s\n",node.name());
				return -1;
			}
                }
		return 0;
	}


int GenericAction::Unstack () {
		while(stack--) {
			solver->hands.pop_back();
		}
		return 0;
	}


int GenericAction::Finish () {
		if (stack > 0) {
			WARNING("Generic action still stacked at finish\n");
			Unstack();
		}
		return 0;
	}


