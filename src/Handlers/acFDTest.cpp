#include "acFDTest.h"
std::string acFDTest::xmlname = "FDTest";
#include "../HandlerFactory.h"

int acFDTest::OptimizerInit () {
		start = NULL;
       		if (Pars == 0) {
       			ERROR("Error: No parameters defined!\n");
       			return -1;
       		}			
		notice("Parameters in test: %d\n", Pars);
		pugi::xml_attribute attr = node.attribute("order");
		if (attr) {
			order = attr.as_int();
		} else {
			order = 2;
		}
		if (order > 6) {
			ERROR("Too high order in FDTest");
			order = 6;
		}
		if (order < 2) {
			ERROR("Negative order in FDTest");
			order = 2;
		}
		order = order + (order % 2);
		order = order / 2;
		
		attr = node.attribute("parameters");
		par_start = 0;
		par_num = Pars;
		if (attr) {
			std::string par = attr.value();
                        unsigned int i = par.find_first_of(':');
                        if (i == string::npos) {
				par_start = attr.as_int();
                        } else {
				if (i == 0) {
					par_num = atoi(par.substr(i+1).c_str())+1;
				} else if (i == par.size()-1) {
					par_start = atoi(par.substr(0,i).c_str());
					par_num = Pars - par_start;
				} else {
					par_start = atoi(par.substr(0,i).c_str());
					par_num = atoi(par.substr(i+1).c_str()) - par_start + 1;
				}
			}			
		}
		output("Parameters in test: %d:%d (%d) from %d\n",par_start,par_start+par_num-1,par_num,Pars);
		if (par_num < 1) { ERROR("You need at least one parameter in FDTest"); return -1; }
		if (par_start < 0) { ERROR("You need at least one parameter in FDTest"); return -1; }
		if ((par_start < 0) || (par_start >= Pars) || (par_start+par_num > Pars)) {
			ERROR("Parameter out of bounds in FDTest");
			return -1;
		}

		h_min = log(1e-11);
		attr = node.attribute("hmin");
		if (attr) {
			double h = attr.as_double();
			if (h <= 0) { ERROR("You need to provide a H > 0 in FDTest"); return -1; }
			h_min = log(h);
		}
		h_max = log(1);
		attr = node.attribute("hmax");
		if (attr) {
			double h = attr.as_double();
			if (h <= 0) { ERROR("You need to provide a H > 0 in FDTest"); return -1; }
			h_max = log(h);
		}
		h_n = 24;
		attr = node.attribute("levels");
		if (attr) {
			h_n = attr.as_int();
			if (h_n < 2) { ERROR("You need to provide a levels >= 2 in FDTest"); return -1; }
		}
		output("Levels is test: %le .. %le (%d)\n",exp(h_min), exp(h_max), h_n);
		for (int ih=0; ih<h_n; ih++) {
			double h = exp(h_min + ((h_max - h_min)*ih)/(h_n-1));
			notice("H Level: %le\n", h);
		}


		start = new double[Pars];
		grad = new double[Pars];
		lower = new double[Pars];
		upper = new double[Pars];
		dx = new double[Pars];
		x = new double[Pars];
		DEBUG_M;
		GetParameters(start);
		Parameters(PAR_LOWER, lower);
		Parameters(PAR_UPPER, upper);
		for (int i=0;i<Pars; i++) dx[i] = (upper[i]-lower[i])/2.0;
		DEBUG_M;
		return 0;
	}


int acFDTest::OptimizerRun () {
		double val0;
		output("Evalulation for testing point");
		for (int i=0;i<Pars; i++) x[i] = start[i];
		val0 = FOptimize(Pars, x, grad, this);
		FILE * f;
		f = fopen((std::string(solver->info.outpath) + "_FD_test.csv").c_str(),"w");
		fprintf(f, "Parameter, Value, Gradient, H");
		for (int o=order; o > 0; o--) fprintf(f,", RightObj%d",o);
		fprintf(f, ", Objective");
		for (int o=0; o < order; o++) fprintf(f,", LeftObj%d",o+1);
		for (int o=order; o >0; o--) fprintf(f,", CentralDiff%d",o);
		fprintf(f,"\n");
		for (int k=par_start; k<par_start+par_num; k++) {
			output("Testing parameter %d",k);
			for (int ih=0; ih<h_n; ih++) {
				double h = dx[k] * exp(h_min + ((h_max - h_min)*ih)/(h_n-1));
				output("Running h=%le\n",h);
				for (int i=0;i<Pars; i++) x[i] = start[i];
				double val[9]; const int p=4;
				for (int m=-order; m<=order;m++) if(m != 0) {
					x[k] = start[k] + m*h;
					val[p+m] = FOptimize(Pars, x, NULL, this);
				} else {
					val[p+m] = val0;
				}
				fprintf(f,"%d, %.16lg, %.16lg, %.16lg", k, start[k], grad[k], h);
				for (int m=-order; m<=order;m++) fprintf(f,", %.16lg", val[p+m]);
				double diff;
				switch(order) {
				case 3:
					diff = (val[p+3] + 9*(-val[p+2] + 5*(val[p+1] - val[p-1]) + val[p-2]) - val[p-3])/60;
					fprintf(f,", %.16lg", diff/h);
				case 2:
					diff = (-val[p+2] + 8*(val[p+1] - val[p-1]) + val[p-2])/12;
					fprintf(f,", %.16lg", diff/h);
				case 1:
					diff = (val[p+1] - val[p-1])/2;
					fprintf(f,", %.16lg", diff/h);
				}
				fprintf(f,"\n");
				fflush(f);
			}
		}
		fclose(f);
		return 0;
	}


int acFDTest::OptimizerExit () {
		return 0;
	}


// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register< GenericAsk< acFDTest > >;
