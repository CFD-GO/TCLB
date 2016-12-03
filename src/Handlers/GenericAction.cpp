#include "GenericAction.h"

int GenericAction::Init () {
		stack=0;
		parSize= -1;
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
						hand.DoIt();
					}
				}
			} else return -1;
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


int GenericAction::NumberOfParameters () {
		output("Collecting parameters from Design elements\n");
		if (parSize < 0) {
			parSize = 0;
			for (size_t i=0; i<solver->hands.size(); i++) if (solver->hands[i].Type()  == HANDLER_DESIGN) {
				output("Getting number of parameters from %s\n", solver->hands[i]->node.name());
				int k = solver->hands[i]->NumberOfParameters();
				parSize += k;
			}
		} else {
			output("Done some time ago\n");
		}
		return parSize;
	};


int GenericAction::Parameters (int type, double * tab) {
		int offset = 0, size = 0, ret=0;
		for (size_t i=0; i<solver->hands.size(); i++) if (solver->hands[i].Type()  == HANDLER_DESIGN) {
			output("Parameters from %s (%d)\n", solver->hands[i]->node.name(), type);
			size = solver->hands[i]->NumberOfParameters();
			if (offset + size > parSize) { offset = offset + size; break; }
			ret = solver->hands[i]->Parameters(type, tab+offset);
			if (ret) return ret;
			offset += size;
		}
		if (offset != parSize) {
				ERROR("Numer of parameters is inconsistent with first call to NumberOfParameters (in Parameters(%d)!", type);
				exit(-1);
				return -1;
		}
		return 0;
	};


