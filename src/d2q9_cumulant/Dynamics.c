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
#define pi 3.141592653589793116


CudaDeviceFunction real_t getRho(){
	return f[8] + f[7] + f[6] + f[5] + f[4] + f[3] + f[2] + f[1] + f[0] ;
}
    
CudaDeviceFunction vector_t getU(){
	real_t d = f[8] + f[7] + f[6] + f[5] + f[4] + f[3] + f[2] + f[1] + f[0];
	vector_t u;
    u.x = f[8] - f[7] - f[6] + f[5] - f[3] + f[1];
    u.y = -f[8] - f[7] + f[6] + f[5] - f[4] + f[2];
    u.x += ForceX * 0.5;
    u.y += ForceY * 0.5; 
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

 real_t c[9],d;
 d = getRho();
 real_t  w[5] = {1.0/(3*nu+0.5),1.,1.,1.,1.0};  // defining relaxation rate for first cummulants
 if ((NodeType & NODE_BOUNDARY) != 0) w[0] = 1.0/(3*nubuffer+0.5);

//First determing moments from density-probability function
  
  f[0] = f[3] + f[1] + f[0]; 
  f[1] = -f[3] + f[1];
  f[3] = f[1] + f[3]*2.; 
  f[2] = f[6] + f[5] + f[2];
  f[5] = -f[6] + f[5];
  f[6] = f[5] + f[6]*2.; 
  f[4] = f[7] + f[8] + f[4];
  f[8] = -f[7] + f[8];
  f[7] = f[8] + f[7]*2.; 
  f[0] = f[4] + f[2] + f[0];
  f[2] = -f[4] + f[2];
  f[4] = f[2] + f[4]*2.; 
  f[1] = f[8] + f[5] + f[1];
  f[5] = -f[8] + f[5];
  f[8] = f[5] + f[8]*2.; 
  f[3] = f[7] + f[6] + f[3];
  f[6] = -f[7] + f[6];
  f[7] = f[6] + f[7]*2.; 
  
//Cumulant calculation from moments
  c[0] = f[0];
  c[1] = f[1]/f[0];
  c[3] = ( -c[1]*f[1] + f[3] )/f[0];
  c[2] = f[2]/f[0];
  c[5] = ( -c[1]*f[2] + f[5] )/f[0];
  c[6] = ( -c[5]*f[1] - c[3]*f[2] - c[1]*f[5] + f[6] )/f[0];
  c[4] = ( -c[2]*f[2] + f[4] )/f[0];
  c[8] = ( -c[1]*f[4] + f[8] - c[5]*f[2]*2. )/f[0];
  c[7] = ( -c[8]*f[1] - c[3]*f[4] - c[1]*f[8] + f[7] + ( -c[6]*f[2] - c[5]*f[5] )*2. )/f[0];
//Cumulant relaxation:
 real_t  a = (c[3] + c[4]);
 real_t  b = (c[3] - c[4]);

//Forcing
  c[1] = c[1] + ForceX;
  c[2] = c[2] + ForceY;
//END Forcing
 
 //real_t Dxu = - w[0]*(2*c[3] - c[4])/(2.*d) - w[1]*(c[3] + c[4])/d;
 //real_t Dyv =  - w[0]*(2*c[4] - c[3])/(2.*d) - w[1]*(c[3] + c[4])/d;
// c[1] = -c[1];
  // c[3] = (1 - w[0])*c[3] + w[0]*1./3;
   c[3] = ((1 - w[1])*a + w[1]*2./3. + (1 - w[0])*b)/2.;
// c[2] =-c[2];
  // c[4] = (1 - w[0])*c[4] + w[0]*1./3;
   c[4] = ((1 - w[1])*a + w[1]*2./3. - (1 - w[0])*b)/2.;
 c[5] =  (1- w[0])*c[5];
 c[6] =  (1 - w[2])*c[6];
 c[7] =  (1 - w[3])*c[7];
 c[8] = (1 - w[2])*c[8]; 


// Moment calculation from cummulants

  f[0] = f[0];
  f[1] = c[1]*f[0];
  f[3] = c[3]*f[0] + c[1]*f[1];
  f[2] = c[2]*f[0];
  f[5] = c[5]*f[0] + c[1]*f[2];
  f[6] = c[6]*f[0] + c[5]*f[1] + c[3]*f[2] + c[1]*f[5];
  f[4] = c[4]*f[0] + c[2]*f[2];
  f[8] = c[8]*f[0] + c[1]*f[4] + c[5]*f[2]*2.;
  f[7] = c[7]*f[0] + c[8]*f[1] + c[3]*f[4] + c[1]*f[8] + ( c[6]*f[2] + c[5]*f[5] )*2.;
 
 //Transformation from moment to density distribution function

  f[0] = -f[3] + f[0];
  f[1] = ( f[3] + f[1] )/2.;
  f[3] = f[3] - f[1]; 
  f[2] = -f[6] + f[2];
  f[5] = ( f[6] + f[5] )/2.;
  f[6] = f[6] - f[5]; 
  f[4] = -f[7] + f[4];
  f[8] = ( f[7] + f[8] )/2.;
  f[7] = f[7] - f[8]; 
  f[0] = -f[4] + f[0];
  f[2] = ( f[4] + f[2] )/2.;
  f[4] = f[4] - f[2]; 
  f[1] = -f[8] + f[1];
  f[5] = ( f[8] + f[5] )/2.;
  f[8] = f[8] - f[5]; 
  f[3] = -f[7] + f[3];
  f[6] = ( f[7] + f[6] )/2.;
  f[7] = f[7] - f[6]; 

}
