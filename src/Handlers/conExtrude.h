#ifndef BSPLINE_H
#define BSPLINE_H

#include "../CommonHandler.h"

#include "vHandler.h"
#include "Callback.h"
#include "Design.h"

class  conExtrude  : public  Design  {
	size_t Pars;
	size_t Pars2;
	int direction;
	double * coords[4];
	double * tab2;
	double * Par;
	double theta;
	double margin;
	std::vector<int> idx;
	Handler * hand;
	bool next(size_t i);
	double Fun(double,double);	
	double FunD(double,double);	
public:
	static std::string xmlname;
int Init ();
int Finish ();
int NumberOfParameters ();
double Pos (int j);
int Parameters (int type, double * tab);
};

#endif // BSPLINE_H
