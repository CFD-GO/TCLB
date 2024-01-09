#ifndef PARTICLE_HPP
#define PARTICLE_HPP

#define NO_SYNC 0x01
#define WARP_SYNC 0x02
#define BLOCK_SYNC 0x03
#define OPP_SYNC 0x04

struct Particle {
    real_t* particle_data;
	vector_t cvel, diff;
	real_t rad; real_t dist;
	CudaDeviceFunction bool in() {
		return dist < rad;
	}
};

struct ParticleI : Particle {
	size_t i;
	CudaDeviceFunction ParticleI(real_t* particle_data_, const size_t& i_, const real_t node[3]) : i(i_) {
        particle_data = particle_data_;
		vector_t pos, vel, angvel;
		rad = particle_data[i*RFI_DATA_SIZE+RFI_DATA_R];
		pos.x = particle_data[i*RFI_DATA_SIZE+RFI_DATA_POS+0];
		pos.y = particle_data[i*RFI_DATA_SIZE+RFI_DATA_POS+1];
		pos.z = particle_data[i*RFI_DATA_SIZE+RFI_DATA_POS+2];
		vel.x = particle_data[i*RFI_DATA_SIZE+RFI_DATA_VEL+0];
		vel.y = particle_data[i*RFI_DATA_SIZE+RFI_DATA_VEL+1];
		vel.z = particle_data[i*RFI_DATA_SIZE+RFI_DATA_VEL+2];
		angvel.x = particle_data[i*RFI_DATA_SIZE+RFI_DATA_ANGVEL+0];
		angvel.y = particle_data[i*RFI_DATA_SIZE+RFI_DATA_ANGVEL+1];
		angvel.z = particle_data[i*RFI_DATA_SIZE+RFI_DATA_ANGVEL+2];
		diff.x = pos.x - node[0];
		diff.y = pos.y - node[1];
		diff.z = pos.z - node[2];
		dist = sqrt(diff.x*diff.x + diff.y*diff.y + diff.z*diff.z);
		cvel.x = vel.x + angvel.y*diff.z - angvel.z*diff.y;
		cvel.y = vel.y + angvel.z*diff.x - angvel.x*diff.z;
		cvel.z = vel.z + angvel.x*diff.y - angvel.y*diff.x;
	}
};

template <int SYNC>
struct ParticleS : ParticleI {
	vector_t force;
	vector_t moment;
	CudaDeviceFunction ParticleS(real_t* particle_data_, const size_t& i_, const real_t node[3]) : ParticleI(particle_data_, i_, node) {
		force.x = 0;
		force.y = 0;
		force.z = 0;
		moment.x = 0;
		moment.y = 0;
		moment.z = 0;
	};
	CudaDeviceFunction inline void applyForce(vector_t f) {
		force.x += f.x;
		force.y += f.y;
		force.z += f.z;
		moment.x -= f.y*diff.z - f.z*diff.y;
		moment.y -= f.z*diff.x - f.x*diff.z;
		moment.z -= f.x*diff.y - f.y*diff.x;
	}
	CudaDeviceFunction ~ParticleS();
};

template <>
CudaDeviceFunction ParticleS< NO_SYNC >::~ParticleS() {
	CudaAtomicAdd(&particle_data[i*RFI_DATA_SIZE+RFI_DATA_FORCE+0],force.x);
	CudaAtomicAdd(&particle_data[i*RFI_DATA_SIZE+RFI_DATA_FORCE+1],force.y);
	CudaAtomicAdd(&particle_data[i*RFI_DATA_SIZE+RFI_DATA_FORCE+2],force.z);
	CudaAtomicAdd(&particle_data[i*RFI_DATA_SIZE+RFI_DATA_MOMENT+0],moment.x);
	CudaAtomicAdd(&particle_data[i*RFI_DATA_SIZE+RFI_DATA_MOMENT+1],moment.y);
	CudaAtomicAdd(&particle_data[i*RFI_DATA_SIZE+RFI_DATA_MOMENT+2],moment.z);
}

template <>
CudaDeviceFunction ParticleS< WARP_SYNC >::~ParticleS() {
	real_t val[6] = {force.x,force.y,force.z,moment.x,moment.y,moment.z};
	CudaAtomicAddReduceWarpArr<6>(&particle_data[i*RFI_DATA_SIZE+RFI_DATA_FORCE],val);
}

#ifdef CROSS_HAS_ADDOPP
template <>
CudaDeviceFunction ParticleS< OPP_SYNC >::~ParticleS() {
	real_t val[6] = {force.x,force.y,force.z,moment.x,moment.y,moment.z};
	CudaAtomicAddReduceOppArr<6>(&particle_data[i*RFI_DATA_SIZE+RFI_DATA_FORCE],val);
}
#endif


template <>
CudaDeviceFunction ParticleS< BLOCK_SYNC >::~ParticleS() {
	CudaAtomicAddReduce(&particle_data[i*RFI_DATA_SIZE+RFI_DATA_FORCE+0],force.x);
	CudaAtomicAddReduce(&particle_data[i*RFI_DATA_SIZE+RFI_DATA_FORCE+1],force.y);
	CudaAtomicAddReduce(&particle_data[i*RFI_DATA_SIZE+RFI_DATA_FORCE+2],force.z);
	CudaAtomicAddReduce(&particle_data[i*RFI_DATA_SIZE+RFI_DATA_MOMENT+0],moment.x);
	CudaAtomicAddReduce(&particle_data[i*RFI_DATA_SIZE+RFI_DATA_MOMENT+1],moment.y);
	CudaAtomicAddReduce(&particle_data[i*RFI_DATA_SIZE+RFI_DATA_MOMENT+2],moment.z);
}


#ifdef USE_ADDOPP
	#define PART_SYNC OPP_SYNC
#else
	#define PART_SYNC WARP_SYNC
#endif

#ifdef SOLID_CACHE
	typedef typename solidcontainer_t::cache_set_found_t< ParticleS< PART_SYNC >, SOLID_CACHE > set_found_t;
    typedef typename solidcontainer_t::cache_set_found_t< ParticleS< PART_SYNC >, SOLID_CACHE > set_found_t_s;
	typedef typename solidcontainer_t::cache_set_found_t< ParticleI, SOLID_CACHE > set_found_t_i;
#else
	typedef typename solidcontainer_t::set_found_t< ParticleS< PART_SYNC > > set_found_t;
    typedef typename solidcontainer_t::set_found_t< ParticleS< PART_SYNC > > set_found_t_s;
    typedef typename solidcontainer_t::set_found_t< ParticleI > set_found_t_i;
#endif

#endif // PARTICLE_HPP
