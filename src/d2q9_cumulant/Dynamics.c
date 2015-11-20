/*-------------------------------------------------------------*/
/*  CLB - Cudne LB - Stencil Version                           */
/*     CUDA based Adjoint Lattice Boltzmann Solver             */
/*     Author: Lukasz Laniewski-Wollk                          */
/*     Developed at: Warsaw University of Technology - 2012    */
/*-------------------------------------------------------------*/


#define S2 1.3333
#define S3 1.0
#define S5 1.0
#define S7 1.0
#define S8 omega
#define S9 omega


CudaDeviceFunction real_t getRho(){
	return f[8] + f[7] + f[6] + f[5] + f[4] + f[3] + f[2] + f[1] + f[0] ;
}
    
CudaDeviceFunction vector_t getU(){
	real_t d = f[8] + f[7] + f[6] + f[5] + f[4] + f[3] + f[2] + f[1] + f[0];
	vector_t u;
u.x = f[8] - f[7] - f[6] + f[5] - f[3] + f[1];
u.y = -f[8] - f[7] + f[6] + f[5] - f[4] + f[2];

	u.x /= d;
	u.y /= d;
	u.z = 0.0;
	return u;
}

CudaDeviceFunction float2 Color() {
        float2 ret;
        vector_t u = getU();
        ret.x = sqrt(u.x*u.x + u.y*u.y);
        if (NodeType == NODE_Solid){
                ret.y = 0;
        } else {
                ret.y = 1;
        }
        return ret;
}

CudaDeviceFunction void SetEquilibrum(real_t d, real_t u[2])
{
f[0] = ( 2. + ( -u[1]*u[1] - u[0]*u[0] )*3. )*d*2./9.;
f[1] = ( 2. + ( -u[1]*u[1] + ( 1 + u[0] )*u[0]*2. )*3. )*d/18.;
f[2] = ( 2. + ( -u[0]*u[0] + ( 1 + u[1] )*u[1]*2. )*3. )*d/18.;
f[3] = ( 2. + ( -u[1]*u[1] + ( -1 + u[0] )*u[0]*2. )*3. )*d/18.;
f[4] = ( 2. + ( -u[0]*u[0] + ( -1 + u[1] )*u[1]*2. )*3. )*d/18.;
f[5] = ( 1 + ( ( 1 + u[1] )*u[1] + ( 1 + u[0] + u[1]*3. )*u[0] )*3. )*d/36.;
f[6] = ( 1 + ( ( 1 + u[1] )*u[1] + ( -1 + u[0] - u[1]*3. )*u[0] )*3. )*d/36.;
f[7] = ( 1 + ( ( -1 + u[1] )*u[1] + ( -1 + u[0] + u[1]*3. )*u[0] )*3. )*d/36.;
f[8] = ( 1 + ( ( -1 + u[1] )*u[1] + ( 1 + u[0] - u[1]*3. )*u[0] )*3. )*d/36.;

}

CudaDeviceFunction void Init() {
	real_t u[2] = {Velocity,0.};
	real_t d = Density;
	SetEquilibrum(d,u);
}

CudaDeviceFunction void Run() {
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
    }
    if (NodeType & NODE_MRT)
    {
		CollisionMRT();
    }
}

CudaDeviceFunction void BounceBack()
{
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

CudaDeviceFunction void CollisionMRT()
{
	real_t u[2], usq, d, R[6];
d = f[8] + f[7] + f[6] + f[5] + f[4] + f[3] + f[2] + f[1] + f[0];
u[0] = f[8] - f[7] - f[6] + f[5] - f[3] + f[1];
u[1] = -f[8] - f[7] + f[6] + f[5] - f[4] + f[2];
R[0] = -f[4] - f[3] - f[2] - f[1] + ( f[8] + f[7] + f[6] + f[5] - f[0]*2 )*2;
R[1] = f[8] + f[7] + f[6] + f[5] + ( -f[4] - f[3] - f[2] - f[1] + f[0]*2 )*2;
R[2] = f[8] - f[7] - f[6] + f[5] + ( f[3] - f[1] )*2;
R[3] = -f[8] - f[7] + f[6] + f[5] + ( f[4] - f[2] )*2;
R[4] = -f[4] + f[3] - f[2] + f[1];
R[5] = -f[8] + f[7] - f[6] + f[5];
usq = u[1]*u[1] + u[0]*u[0];

switch (NodeType & NODE_OBJECTIVE) {
case NODE_Outlet:
	AddToOutletFlux(u[0]/d);
	AddToPressureLoss(-u[0]/d*((d-1.)/3. + usq/d/2.));
	break;
case NODE_Inlet:
	AddToInletFlux(u[0]/d);
	AddToPressureLoss(u[0]/d*((d-1.)/3. + usq/d/2.));
	break;
}
R[0] = R[0]*(1-S2)  +  S2*(-2. * d + 3. * usq);
R[1] = R[1]*(1-S3)  +  S3*(d - 3.*usq);
R[2] = R[2]*(1-S5)  +  S5*(-u[0]);
R[3] = R[3]*(1-S7)  +  S7*(-u[1]);
R[4] = R[4]*(1-S8)  +  S8*(u[0]*u[0] - u[1]*u[1]);
R[5] = R[5]*(1-S9)  +  S9*(u[0]*u[1]);

d /= 9; 
u[0] /= 6; 
u[1] /= 6; 
R[0] /= 36; 
R[1] /= 36; 
R[2] /= 12; 
R[3] /= 12; 
R[4] /= 4; 
R[5] /= 4; 
f[0] = d + ( R[1] - R[0] )*4;
f[1] = R[4] - R[0] + u[0] + d + ( -R[2] - R[1] )*2;
f[2] = -R[4] - R[0] + u[1] + d + ( -R[3] - R[1] )*2;
f[3] = R[4] - R[0] - u[0] + d + ( R[2] - R[1] )*2;
f[4] = -R[4] - R[0] - u[1] + d + ( R[3] - R[1] )*2;
f[5] = R[5] + R[3] + R[2] + R[1] + u[1] + u[0] + d + R[0]*2;
f[6] = -R[5] + R[3] - R[2] + R[1] + u[1] - u[0] + d + R[0]*2;
f[7] = R[5] - R[3] - R[2] + R[1] - u[1] - u[0] + d + R[0]*2;
f[8] = -R[5] - R[3] + R[2] + R[1] - u[1] + u[0] + d + R[0]*2;

}
