struct Particle {
//	vector_t pos, vel, angvel;
	vector_t cvel, diff;
	real_t rad; real_t dist;
	CudaDeviceFunction bool in() {
		return dist < rad;
	}
};

#define safe_push_all if (P::valid_()) this->push_all
#define safe_pull_all if (P::valid_()) this->pull_all
#define safe_pull_base if (P::valid_()) this->pull_base
#define safe_pull_rest if (P::valid_()) this->pull_rest

struct ParticleI : Particle {
	static const bool sync = false;
	static CudaDeviceFunction inline bool SyncOr(const bool& b) { return b; }
	size_t i;
	real_t node[3];
	CudaDeviceFunction inline bool valid_() { return i < constContainer.particle_data_size; }
	CudaDeviceFunction operator bool () { return valid_(); }
	CudaDeviceFunction ParticleI(real_t x, real_t y, real_t z) { node[0] = x; node[1] = y; node[2] = z; i = 0; };
	CudaDeviceFunction void push_all() {
	}
	CudaDeviceFunction void pull_all() {
		vector_t pos, vel, angvel;
		rad = constContainer.particle_data[i*RFI_DATA_SIZE+RFI_DATA_R];
		pos.x = constContainer.particle_data[i*RFI_DATA_SIZE+RFI_DATA_POS+0];
		pos.y = constContainer.particle_data[i*RFI_DATA_SIZE+RFI_DATA_POS+1];
		pos.z = constContainer.particle_data[i*RFI_DATA_SIZE+RFI_DATA_POS+2];
		vel.x = constContainer.particle_data[i*RFI_DATA_SIZE+RFI_DATA_VEL+0];
		vel.y = constContainer.particle_data[i*RFI_DATA_SIZE+RFI_DATA_VEL+1];
		vel.z = constContainer.particle_data[i*RFI_DATA_SIZE+RFI_DATA_VEL+2];
		angvel.x = constContainer.particle_data[i*RFI_DATA_SIZE+RFI_DATA_ANGVEL+0];
		angvel.y = constContainer.particle_data[i*RFI_DATA_SIZE+RFI_DATA_ANGVEL+1];
		angvel.z = constContainer.particle_data[i*RFI_DATA_SIZE+RFI_DATA_ANGVEL+2];
		diff.x = pos.x - node[0];
		diff.y = pos.y - node[1];
		diff.z = pos.z - node[2];
		dist = sqrt(diff.x*diff.x + diff.y*diff.y + diff.z*diff.z);
		cvel.x = vel.x + angvel.y*diff.z - angvel.z*diff.y;
		cvel.y = vel.y + angvel.z*diff.x - angvel.x*diff.z;
		cvel.z = vel.z + angvel.x*diff.y - angvel.y*diff.x;
	}
};

struct ParticleS : ParticleI {
	static const bool sync = true;
	static CudaDeviceFunction inline bool SyncOr(const bool& b) { return CudaSyncThreadsOr(b); }
	vector_t force;
	vector_t moment;
	CudaDeviceFunction ParticleS(real_t x, real_t y, real_t z) : ParticleI(x,y,z) {};
	CudaDeviceFunction inline void applyForce(vector_t f) {
		force.x += f.x;
		force.y += f.y;
		force.z += f.z;
		moment.x -= f.y*diff.z - f.z*diff.y;
		moment.y -= f.z*diff.x - f.x*diff.z;
		moment.z -= f.x*diff.y - f.y*diff.x;
	}
	CudaDeviceFunction void push_all() {
		atomicSum(&constContainer.particle_data[i*RFI_DATA_SIZE+RFI_DATA_FORCE+0],force.x);
		atomicSum(&constContainer.particle_data[i*RFI_DATA_SIZE+RFI_DATA_FORCE+1],force.y);
		atomicSum(&constContainer.particle_data[i*RFI_DATA_SIZE+RFI_DATA_FORCE+2],force.z);
		atomicSum(&constContainer.particle_data[i*RFI_DATA_SIZE+RFI_DATA_MOMENT+0],moment.x);
		atomicSum(&constContainer.particle_data[i*RFI_DATA_SIZE+RFI_DATA_MOMENT+1],moment.y);
		atomicSum(&constContainer.particle_data[i*RFI_DATA_SIZE+RFI_DATA_MOMENT+2],moment.z);
		ParticleI::push_all();
	}
	CudaDeviceFunction void pull_all() {
		ParticleI::pull_all();
		force.x = 0;
		force.y = 0;
		force.z = 0;
		moment.x = 0;
		moment.y = 0;
		moment.z = 0;
	}
};

template < class P >
struct AllParticleIterator : P {
	CudaDeviceFunction AllParticleIterator(real_t x, real_t y, real_t z) : P(x,y,z) {
		safe_pull_all();
	}
	CudaDeviceFunction void operator++ () {
		safe_push_all();
		this->i++;
		safe_pull_all();
	}
};

template < class P >
struct TreeParticleIterator : P {
	tr_addr_t nodei;
	CudaDeviceFunction inline bool valid_() { return nodei != -1; }
	CudaDeviceFunction operator bool () { return valid_(); }
	CudaDeviceFunction TreeParticleIterator(real_t x, real_t y, real_t z) : P(x,y,z) {
	        nodei = 0;
	        if (constContainer.particle_data_size == 0) {
	        	nodei = -1;
	        	return;
		}
	        go(true);
		safe_pull_all();
	}
	CudaDeviceFunction void go(bool go_left) {
		while (nodei != -1) {
		    tr_elem elem = constContainer.balltree_data[nodei];
		    if (elem.flag >= 4) { this->i = elem.right; break; }
		    int dir = elem.flag;
		    if (go_left) if (P::SyncOr(this->node[dir] < elem.b)) { nodei++; continue; }
		    go_left = true;
		    if (P::SyncOr(this->node[dir] >= elem.a)) { nodei = elem.right; continue; }
		    go_left = false;
		    nodei = elem.back;
		}    
	}
	CudaDeviceFunction void operator++ () {
		if (nodei != -1) {
		    this->push_all();
		    nodei = constContainer.balltree_data[nodei].back;
		    go(false);
		}
	        safe_pull_all();
	}
};


typedef TreeParticleIterator< ParticleS > SyncParticleIterator;
typedef TreeParticleIterator< ParticleI > ParticleIterator;

typedef AllParticleIterator< ParticleI > FullParticleIterator;
typedef AllParticleIterator< ParticleS > SyncFullParticleIterator;

/*template <class N> CudaDeviceFunction void FillParticle(N& now, Particle& p, size_t& i) {
	p.rad = constContainer.particle_data[i*RFI_DATA_SIZE+RFI_DATA_R];
	p.pos.x = constContainer.particle_data[i*RFI_DATA_SIZE+RFI_DATA_POS+0];
	p.pos.y = constContainer.particle_data[i*RFI_DATA_SIZE+RFI_DATA_POS+1];
	p.pos.z = constContainer.particle_data[i*RFI_DATA_SIZE+RFI_DATA_POS+2];
	p.vel.x = constContainer.particle_data[i*RFI_DATA_SIZE+RFI_DATA_VEL+0];
	p.vel.y = constContainer.particle_data[i*RFI_DATA_SIZE+RFI_DATA_VEL+1];
	p.vel.z = constContainer.particle_data[i*RFI_DATA_SIZE+RFI_DATA_VEL+2];
	p.angvel.x = constContainer.particle_data[i*RFI_DATA_SIZE+RFI_DATA_ANGVEL+0];
	p.angvel.y = constContainer.particle_data[i*RFI_DATA_SIZE+RFI_DATA_ANGVEL+1];
	p.angvel.z = constContainer.particle_data[i*RFI_DATA_SIZE+RFI_DATA_ANGVEL+2];
	p.force.x = 0;
	p.force.y = 0;
	p.force.z = 0;
	p.moment.x = 0;
	p.moment.y = 0;
	p.moment.z = 0;
	p.diff.x = p.pos.x - now.x_ - constContainer.px;
	p.diff.y = p.pos.y - now.y_ - constContainer.py;
	p.diff.z = p.pos.z - now.z_ - constContainer.pz;
	p.dist = sqrt(p.diff.x*p.diff.x + p.diff.y*p.diff.y + p.diff.z*p.diff.z);
	p.cvel.x = p.vel.x + p.angvel.y*p.diff.z - p.angvel.z*p.diff.y;
	p.cvel.y = p.vel.y + p.angvel.z*p.diff.x - p.angvel.x*p.diff.z;
	p.cvel.z = p.vel.z + p.angvel.x*p.diff.y - p.angvel.y*p.diff.x;
}
*/
