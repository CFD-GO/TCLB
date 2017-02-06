#ifndef VHANDLER_H
#define VHANDLER_H

#include "../Consts.h"
#include "../pugixml.hpp"
#include <math.h>
#define HANDLER_CALLBACK  0x01
#define HANDLER_ACTION    0x02
#define HANDLER_DESIGN    0x04
#define HANDLER_GENERIC   0x10
#define HANDLER_CONTAINER 0x20

#define PAR_GET   0x01
#define PAR_SET   0x02
#define PAR_GRAD  0x03
#define PAR_LOWER 0x04
#define PAR_UPPER 0x05

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
	virtual ~vHandler() {};
	virtual int Init(); ///< Initialize the Handler
	virtual int DoIt(); ///< Do what have to be done
	virtual int Finish(); ///< Finalize the Handler
	virtual int Type(); ///< Return the type of the Handler
	virtual int NumberOfParameters(); ///< Return the type of the Handler
	virtual int Parameters(int type, double* data);
	inline  int GetParameters(double * data) { return this->Parameters(PAR_GET, data); };
	inline  int SetParameters(const double *data) {return this->Parameters(PAR_SET, const_cast<double *>(data));}; ///< Return the type of the Handler
	inline  int GetGradient(double * data) { return this->Parameters(PAR_GRAD, data); }; ///< Return the type of the Handler
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

#endif // VHANDLER_H