/*-------------------------------------------------------------*/
/*  CLB - Cudne LB - Stencil Version                           */
/*     CUDA based Adjoint Lattice Boltzmann Solver             */
/*     Author: Lukasz Laniewski-Wollk                          */
/*     Developed at: Warsaw University of Technology - 2012    */
/*-------------------------------------------------------------*/

/*
Model created by Travis Mitchell 10-03-2016. Purpose of model is
to give an introduction into model creation in TCLB, tutorial 
file is available upon request.

Model solves d2q9 files via applying the single relaxation time
BGK-lattice Boltzmann method
*/

CudaDeviceFunction float2 Color() {
// used for graphics - can usually ignore function
 /*       float2 ret;
        vector_t u = getU();
        ret.x = sqrt(u.x*u.x + u.y*u.y);
        if (NodeType == NODE_Solid){
                ret.y = 0;
        } else {
                ret.y = 1;
        }
        return ret;*/
}

CudaDeviceFunction void Init() {
// Initialise the velocity at each node 
	real_t u[2] = {Velocity, 0.};
	real_t d    = Density;
	SetEquilibrium(d,u);
}
 
CudaDeviceFunction void Run() {
// This defines the dynamics that we run at each node in the domain.
    switch (NodeType & NODE_BOUNDARY) {
	case NODE_Solid:
	case NODE_Wall:
		BounceBack();
		break;
	case NODE_EVelocity:
		EVelocity();
		break;
	case NODE_WPressure:
		WPressure();
		break;
	case NODE_WVelocity:
		WVelocity();
		break;
	case NODE_EPressure:
		EPressure();
		break;
//	case NODE_NVelocity:
//		NVelocity();
//		break;
    }
	if (NodeType & NODE_MRT) 
	{
	// Set as if MRT as majority of examples specify
	// solution zone as MRT box, so avoid changing 
	// input files.
		CollisionBGK();
	}
}

CudaDeviceFunction void CollisionBGK() {
// Here we perform a single relaxation time collision operation.
// We save memory here by using a single dummy variable

	real_t u[2], d, f_temp[9];
	d = getRho();
	// pu* = pu + rG
	u[0] = (( f[8]-f[7]-f[6]+f[5]-f[3]+f[1] )/d + GravitationX/omega );
	u[1] = ((-f[8]-f[7]+f[6]+f[5]-f[4]+f[2] )/d + GravitationY/omega );
	// feq = f[0]; f new = f_temp[0]
	f_temp[0] = f[0];
	f_temp[1] = f[1];
	f_temp[2] = f[2];
	f_temp[3] = f[3];
	f_temp[4] = f[4];
	f_temp[5] = f[5];
	f_temp[6] = f[6];
	f_temp[7] = f[7];
	f_temp[8] = f[8];
	SetEquilibrium(d, u);
	f[0] = f_temp[0] - omega*(f_temp[0]-f[0]);	
	f[1] = f_temp[1] - omega*(f_temp[1]-f[1]);
	f[2] = f_temp[2] - omega*(f_temp[2]-f[2]);
	f[3] = f_temp[3] - omega*(f_temp[3]-f[3]);	
	f[4] = f_temp[4] - omega*(f_temp[4]-f[4]);
	f[5] = f_temp[5] - omega*(f_temp[5]-f[5]);
	f[6] = f_temp[6] - omega*(f_temp[6]-f[6]);	
	f[7] = f_temp[7] - omega*(f_temp[7]-f[7]);
	f[8] = f_temp[8] - omega*(f_temp[8]-f[8]);
}
CudaDeviceFunction void BounceBack() {
// Method to reverse distribution functions along the bounding nodes.
     	real_t uf;
	uf = f[3];
	f[3] = f[1];
	f[1] = uf;
	uf = f[4];
	f[4] = f[2];
	f[2] = uf;
	uf = f[7];
	f[7] = f[5];
	f[5] = uf;
	uf = f[8];
	f[8] = f[6];
	f[6] = uf;
}
	

CudaDeviceFunction real_t getRho() {
// This function defines the macroscopic density at the current node.
	return f[8]+f[7]+f[6]+f[5]+f[4]+f[3]+f[2]+f[1]+f[0];
}

CudaDeviceFunction vector_t getU() {
// This function defines the macroscopic velocity at the current node.
	real_t d = f[8]+f[7]+f[6]+f[5]+f[4]+f[3]+f[2]+f[1]+f[0];
	vector_t u;
	// pv = pu + G/2
	u.x = (( f[8]-f[7]-f[6]+f[5]-f[3]+f[1] )/d + GravitationX*0.5 );
	u.y = ((-f[8]-f[7]+f[6]+f[5]-f[4]+f[2] )/d + GravitationY*0.5 );
	u.z = 0;
	return u;
}


CudaDeviceFunction void SetEquilibrium(real_t d, real_t u[2])
{
f[0] = ( 2. + ( -u[1]*u[1] - u[0]*u[0] )*3. )*d*2./9.;
f[1] = ( 2. + ( -u[1]*u[1] + ( 1 + u[0] )*u[0]*2. )*3. )*d/18.;
f[2] = ( 2. + ( -u[0]*u[0] + ( 1 + u[1] )*u[1]*2. )*3. )*d/18.;
f[3] = ( 2. + ( -u[1]*u[1] + ( -1 + u[0] )*u[0]*2. )*3. )*d/18.;
f[4] = ( 2. + ( -u[0]*u[0] + ( -1 + u[1] )*u[1]*2. )*3. )*d/18.;
f[5] = ( 1. + ( ( 1 + u[1] )*u[1] + ( 1 + u[0] + u[1]*3. )*u[0] )*3. )*d/36.;
f[6] = ( 1. + ( ( 1 + u[1] )*u[1] + ( -1 + u[0] - u[1]*3. )*u[0] )*3. )*d/36.;
f[7] = ( 1. + ( ( -1 + u[1] )*u[1] + ( -1 + u[0] + u[1]*3. )*u[0] )*3. )*d/36.;
f[8] = ( 1. + ( ( -1 + u[1] )*u[1] + ( 1 + u[0] - u[1]*3. )*u[0] )*3. )*d/36.;
}


CudaDeviceFunction void NVelocity()
{
        real_t rho, ru;
	real_t ux0 = Velocity;
	rho = ( f[0] + f[1] + f[3] + 2.*(f[6] + f[2] + f[5]) ) / (1. + ux0);
	ru = rho * ux0;
	f[4] = f[2] - (2./3.) * ru;
	f[8] = f[6] - (1./6.) * ru + (1./2.)*(f[3] - f[1]);
	f[7] = f[5] - (1./6.) * ru + (1./2.)*(f[1] - f[3]);
}


CudaDeviceFunction void EVelocity()
{
        real_t rho, ru;
	real_t ux0 = Velocity;
	rho = ( f[0] + f[2] + f[4] + 2.*(f[1] + f[5] + f[8]) ) / (1. + ux0);
	ru = rho * ux0;
	f[3] = f[1] - (2./3.) * ru;
	f[7] = f[5] - (1./6.) * ru + (1./2.)*(f[2] - f[4]);
	f[6] = f[8] - (1./6.) * ru + (1./2.)*(f[4] - f[2]);
}

CudaDeviceFunction void WPressure()
{
        real_t ru, ux0;
	real_t rho = Density;
	ux0 = -1. + ( f[0] + f[2] + f[4] + 2.*(f[3] + f[7] + f[6]) ) / rho;
	ru = rho * ux0;

	f[1] = f[3] - (2./3.) * ru;
	f[5] = f[7] - (1./6.) * ru + (1./2.)*(f[4] - f[2]);
	f[8] = f[6] - (1./6.) * ru + (1./2.)*(f[2] - f[4]);
}

CudaDeviceFunction void WVelocity()
{
        real_t rho, ru;
	real_t u[2] = {Velocity,0.};
	rho = ( f[0] + f[2] + f[4] + 2.*(f[3] + f[7] + f[6]) ) / (1. - u[0]);
	ru = rho * u[0];
	f[1] = f[3] + (2./3.) * ru;
	f[5] = f[7] + (1./6.) * ru + (1./2.)*(f[4] - f[2]);
	f[8] = f[6] + (1./6.) * ru + (1./2.)*(f[2] - f[4]);
}

CudaDeviceFunction void EPressure()
{
        real_t ru, ux0;
	real_t rho = Density;
	ux0 = -1. + ( f[0] + f[2] + f[4] + 2.*(f[1] + f[5] + f[8]) ) / rho;
	ru = rho * ux0;

	f[3] = f[1] - (2./3.) * ru;
	f[7] = f[5] - (1./6.) * ru + (1./2.)*(f[2] - f[4]);
	f[6] = f[8] - (1./6.) * ru + (1./2.)*(f[4] - f[2]);
}

