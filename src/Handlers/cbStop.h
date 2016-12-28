#ifndef CBSTOP_H
#define CBSTOP_H

#include "../CommonHandler.h"

#include "vHandler.h"
#include "Callback.h"

class  cbStop  : public  Callback  {
        std::vector< int > what;
        std::vector< double > change;
        std::vector< double > old;
	int times, score;
	int old_iter_type;
	public:
	static std::string xmlname;
int Init ();
int DoIt ();
int Finish ();
};

#endif // CBSTOP_H
