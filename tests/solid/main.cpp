#include <vector>
#include <random>

#include <stdlib.h>
#include <cstring>
#include <algorithm>

#define CROSS_H // We don't want the usual cross facilities, use the mock below
#define GLOBAL_H

#define CudaMalloc(a__,b__) assert( (*((void**)(a__)) = malloc(b__)) != NULL )
#define CudaFree(a__) free(a__)
#define CudaStream_t int
#define CudaSuccess -1
#define CudaMemcpy(a__,b__,c__,d__) memcpy(a__, b__, c__)
#define CudaMemcpyAsync(a__,b__,c__,d__,e__) CudaMemcpy(a__, b__, c__, d__)
#define CudaDeviceFunction
typedef float real_t;

#define ERROR printf
#define output printf

const int max_cache_size = 1024;
#include "SolidTree.h"
#include "SolidAll.h"
#include "SolidGrid.h"


struct ball {
	double pos[3];
	double rad;
};

struct Balls {
	typedef ball particle_t;
	int n;
	ball* balls;
	inline size_t size() { return n; }
	inline double getPos(int i, int j) { return balls[i].pos[j]; }
	inline double getRad(int i) { return balls[i].rad; }
	inline void print(const int& i) {
		printf("Ball[%d]: c:[%lf %lf %lf] r:%lf\n",
			i, balls[i].pos[0], balls[i].pos[1], balls[i].pos[2], balls[i].rad
		);
	}
};


Balls balls;

struct Particle {
	size_t balli;
	double pos[3];
	double rad;
	double dist;
	Particle(const real_t* dummy, int i_, const real_t org[3]): balli(i_) {
		//printf("load Particle... ");
		dist = 0;
		for (int i=0;i<3;i++) {
			pos[i] = org[i] - balls.getPos(balli, i);
			dist += pos[i]*pos[i];
		}
		dist = sqrt(dist);
		rad = balls.getRad(balli);
	}
	void addForce() {
		printf("%lf ...", dist);
	}
	~Particle() {
		//printf("save Particle\n");
	}
};

template <class container_t>
class Tester {
    container_t container;
	typename container_t::finder_t finder;
	public:
	std::string name;
	Tester(const std::string& name_): name(name_) {
		container.balls = &balls;
		printf("Building %s...\n", name.c_str());
		container.Build();
		printf("Done.\n");
		container.InitFinder(finder);
		container.CopyToGPU(finder,0);
	}
	std::vector<int> find(const real_t pos[3], const real_t& offset) {
		std::vector<int> ret;
		typedef typename container_t::set_found_t< Particle > set_found_t;
		real_t lower[3] = {pos[0]-offset,pos[1]-offset,pos[2]-offset};
		real_t upper[3] = {pos[0]+offset,pos[1]+offset,pos[2]+offset};
        for (auto part : set_found_t(finder, nullptr, pos, lower, upper)) {
			if (part.dist <= part.rad+offset) ret.push_back(part.balli);
		}
		std::sort(ret.begin(),ret.end());
		return ret;
	}
	~Tester() {
		container.CleanFinder(finder);
	}
};

class Printer {
	std::set<int> all;
	public:
	void print_idx(const std::vector<int>& idx, const std::string& name) {
		printf("Results from %s:", name.c_str());
		for (int k : idx) {
			printf("%d, ",k);
			all.insert(k);
		}
		printf("\n");
	}
	void print_balls() {
		for (int k : all) {
			if (k >= 0 && ((size_t) k) < balls.size()) {
				balls.print(k);
			} else {
				printf("Out of bounds ball index %d\n", k);
			}
		}
	}
};

int main(int argn, char** argv) { 
	
    int n = 1000;
	if (argn > 1) n = atoi(argv[1]);
    std::uniform_real_distribution<double> pos_dist(0, 128);
    std::uniform_real_distribution<double> rad_dist(4, 8);
	std::uniform_real_distribution<double> point_dist(0, 128);
	std::uniform_real_distribution<double> offset_dist(0, 1);

    std::default_random_engine random_engine;

	balls.balls = new ball[n];
	balls.n = n;
    for (int i=0;i<n;i++) {
        for (int j=0;j<3;j++) balls.balls[i].pos[j] = pos_dist(random_engine);
        balls.balls[i].rad = rad_dist(random_engine);
    }

	Tester< SolidAll < Balls > > test1("All Indexer");
	Tester< SolidTree< Balls > > test2("Tree Indexer");
	Tester< SolidGrid< Balls > > test3("Grid Indexer");

	bool good=true;
	printf("Testing random points and offset...\n");
	printf("[");
	for (int i=0;i<100;i++) {
		real_t pos[3];
		real_t offset;
		for (int j=0;j<3;j++) pos[j] = point_dist(random_engine);
		offset = offset_dist(random_engine);
		
		std::vector<int> idx1 = test1.find(pos,offset);
		std::vector<int> idx2 = test2.find(pos,offset);
		std::vector<int> idx3 = test3.find(pos,offset);
		
		if (idx1 != idx2) good=false;
		if (idx1 != idx3) good=false;
		if (!good) {
			printf("X]\n\n");
			printf("Wrong results, while checking point [%lf %lf %lf] with offset %lf\n", (double)pos[0], (double)pos[1], (double)pos[2], (double) offset);
			Printer p;
			p.print_idx(idx1,test1.name);
			p.print_idx(idx2,test2.name);
			p.print_idx(idx3,test3.name);
			p.print_balls();
			break;
		}
		printf(".");
	}
	delete[] balls.balls;
	if (good) {
		printf("]\n");
		printf("\n## Test success ##\n");
		return 0;
	} else {
		printf("\n## Test failed ##\n");
	}
	return -1;
}
