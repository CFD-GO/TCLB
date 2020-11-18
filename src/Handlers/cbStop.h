#ifndef CBSTOP_H
#define CBSTOP_H

#include "../CommonHandler.h"

#include "vHandler.h"
#include "Callback.h"

#define STOP_CHANGE 0x01
#define STOP_ABOVE 0x02
#define STOP_BELOW 0x03
#define STOP_PERCENTCHANGE 0x04

class  cbStop  : public  Callback  {
        std::vector< int > what;
        std::vector< int > stop_type;
        std::vector< double > limit;
        std::vector< double > old;
	int times, score;
	int old_iter_type;
	public:
	static std::string xmlname;
int Init ();
int DoIt ();
int Finish ();
void AddStop(int what_, int stop_type_, double limit_);
};

#endif // CBSTOP_H
