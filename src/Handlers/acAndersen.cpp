#include "acAndersen.h"
std::string acAndersen::xmlname = "Andersen";
#include "../HandlerFactory.h"


template <typename T>
void rotate(T* tab, int len) {
        if (len < 1) return;
        T last = tab[len-1];
        for (int i = len-1; i > 0; i--) {
                tab[i] = tab[i-1];
        }
        tab[0] = last;
}

double acAndersen::skal(real_t * a, real_t * b) {
        double sum=0;
        for (size_t k=0; k<n; k++) sum += a[k]*b[k];
        // TODO: MPI scatter gather
        double gsum=0;
        MPI_Allreduce ( &sum, &gsum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
        return gsum;
}

int acAndersen::Init () {
                int times;
                double eps = 0;
		GenericAction::Init();
		pugi::xml_attribute attr = node.attribute("Directions");
		if (attr) {
			directions = attr.as_int();
		} else {
			error("no Directions parameter in %s\n",node.name());
			return -1;
		}
		attr = node.attribute("Times");
		if (attr) {
			times = attr.as_int();
		} else {
		        times = directions;
		}
		attr = node.attribute("Eps");
		if (attr) {
			eps = attr.as_double();
		}
		n = solver->getCartLattice()->sizeOfTab();
		output("Size of vector in Andersen: %ld\n", n);
		x = (real_t **) malloc(directions*sizeof(real_t*));
		e = (real_t **) malloc(directions*sizeof(real_t*));
		p = (double *)  malloc(directions*sizeof(double));
		output("Allocating %ld b for Anderson\n",(size_t)2*n*directions * sizeof(real_t));
		real_t * mem = (real_t *) malloc(2*n*directions * sizeof(real_t));
		for (int i = 0; i<directions; i++) {
		        x[i] = &mem[   2*i *n];
		        e[i] = &mem[(2*i+1)*n];
                }
	        real_t * nx = (real_t *) malloc(n * sizeof(real_t));

                int d = 0;

                for (int it = 0; it < times; it++) {
                        rotate(x,directions);
                        rotate(e,directions);
                        rotate(p,directions);
                        
                        
                        solver->getCartLattice()->saveToTab(x[0]);
                        if (GenericAction::ExecuteInternal()) return -1;
                        solver->getCartLattice()->saveToTab(e[0]);
                        for (size_t i=0; i<n; i++) e[0][i] = e[0][i] - x[0][i];
                        double sum = skal(e[0],e[0]);
                        output("Residual in Andersen: %lg\n",sum);
                        if (!isfinite(sum)) break;
                        if (sum < eps) break;
                        p[0] = 1;
                        
                        d++;
                        if (d > directions)
                        	d = directions; // Continue
//                                d = 1;		// Restart

//                        for (int i=0; i < d; i++) {		//Option A
//                        for (int i=d-1; i >= 0; i--) {	//Option B - obsolete
                        { int i=0;				//Option B
//                                for (int j=0; j < i; j++) {	//Option A
                                  for (int j=i+1; j < d; j++) {	//Option B
                                        double a = skal(e[i],e[j]);
                                        debug2("R[%d,%d]: %lg\n",i,j,a);
                                        for (size_t k=0; k<n; k++) e[i][k] -= a * e[j][k];
                                        for (size_t k=0; k<n; k++) x[i][k] -= a * x[j][k];
                                        p[i] -= a * p[j];
                                }
                                double a = sqrt(skal(e[i],e[i]));
                                debug2("R[%d,%d]: %lg\n",i,i,a);
                                for (size_t k=0; k<n; k++) e[i][k] /= a;
                                for (size_t k=0; k<n; k++) x[i][k] /= a;
                                p[i] /= a;
                        }
                        debug2("Andersen p:");
                        for (int i=0; i < d; i++) {
                                debug2(" %lg",p[i]);
                        }
                        debug2("\n");

                        double psum = 0;
                        for (int i=0; i < d; i++) psum += p[i]*p[i];
                        for (size_t k=0; k<n; k++) {
                                double sum = 0;
                                for (int i=0; i < d; i++) sum += x[i][k]*p[i]/psum;
                                nx[k] = sum;
                        }
                        solver->getCartLattice()->loadFromTab(nx);
                        if (GenericAction::ExecuteInternal()) return -1;
                }
                free(mem);
		free(x);
		free(e);
		free(p);
		free(nx);
		return 0;
}


// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register< GenericAsk< acAndersen > >;
