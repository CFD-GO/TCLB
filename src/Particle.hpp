
#define NO_SYNC 0x01
#define WARP_SYNC 0x02
#define BLOCK_SYNC 0x03

#define PARTICLE_SYNC 0x02

struct Particle {
//	vector_t pos, vel, angvel;
	vector_t cvel, diff;
	real_t rad; real_t dist;
	CudaDeviceFunction bool in() {
		return dist < rad;
	}
};

struct ParticleI : Particle {
	size_t i;
	CudaDeviceFunction ParticleI(const size_t& i_, const real_t node[3]): i(i_) {
		// i < constContainer.particle_data_size;
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

template <int SYNC>
struct ParticleS : ParticleI {
	vector_t force;
	vector_t moment;
	CudaDeviceFunction ParticleS(const size_t& i_, const real_t node[3]) : ParticleI(i_,node) {
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
	atomicAddP(&constContainer.particle_data[i*RFI_DATA_SIZE+RFI_DATA_FORCE+0],force.x);
	atomicAddP(&constContainer.particle_data[i*RFI_DATA_SIZE+RFI_DATA_FORCE+1],force.y);
	atomicAddP(&constContainer.particle_data[i*RFI_DATA_SIZE+RFI_DATA_FORCE+2],force.z);
	atomicAddP(&constContainer.particle_data[i*RFI_DATA_SIZE+RFI_DATA_MOMENT+0],moment.x);
	atomicAddP(&constContainer.particle_data[i*RFI_DATA_SIZE+RFI_DATA_MOMENT+1],moment.y);
	atomicAddP(&constContainer.particle_data[i*RFI_DATA_SIZE+RFI_DATA_MOMENT+2],moment.z);
}

template <>
CudaDeviceFunction ParticleS< WARP_SYNC >::~ParticleS() {
	real_t val[6] = {force.x,force.y,force.z,moment.x,moment.y,moment.z};
	atomicSumWarpArr(&constContainer.particle_data[i*RFI_DATA_SIZE+RFI_DATA_FORCE],val,6);
}

template <>
CudaDeviceFunction ParticleS< BLOCK_SYNC >::~ParticleS() {
	atomicSum(&constContainer.particle_data[i*RFI_DATA_SIZE+RFI_DATA_FORCE+0],force.x);
	atomicSum(&constContainer.particle_data[i*RFI_DATA_SIZE+RFI_DATA_FORCE+1],force.y);
	atomicSum(&constContainer.particle_data[i*RFI_DATA_SIZE+RFI_DATA_FORCE+2],force.z);
	atomicSum(&constContainer.particle_data[i*RFI_DATA_SIZE+RFI_DATA_MOMENT+0],moment.x);
	atomicSum(&constContainer.particle_data[i*RFI_DATA_SIZE+RFI_DATA_MOMENT+1],moment.y);
	atomicSum(&constContainer.particle_data[i*RFI_DATA_SIZE+RFI_DATA_MOMENT+2],moment.z);
}

CudaDeviceFunction auto SyncParticleIterator(real_t x, real_t y, real_t z) {
	real_t point[3] = {x,y,z};
	return constContainer.solidfinder.find< ParticleS< WARP_SYNC > >(point, point, point);
//	real_t point[3] = {x,y,z};
//	real_t lower[3] = {x-CudaThread.x,y,z};
//	real_t upper[3] = {x-CudaThread.x+CudaNumberOfThreads.x-1,y,z};
//	return constContainer.solidfinder.find< ParticleS< WARP_SYNC > >(point, lower, upper);
}
