#ifndef CONCONTROL_H
#define CONCONTROL_H

#include "../CommonHandler.h"

#include "vHandler.h"
#include "Action.h"

class  conControl  : public  Action  {
        int iter;
        typedef std::map< std::string , std::vector<double> > Context;
        Context context;
	public:
int Params (pugi::xml_node n);
int get (Context& cont, const char * svar, double scale, std::vector<double>& fill);
int Internal (pugi::xml_node n);
int Init ();
};

#endif // CONCONTROL_H
