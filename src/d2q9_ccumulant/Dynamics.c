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
        real_t rho, ux, uy;
        rho = (1+Pressure*3);
        if (SL_L > 0) {
                if (Y < SL_L/2) {
                        ux = SL_U * tanh(SL_lambda * ( Y/SL_L - 0.25 ));
                } else {
                        ux = SL_U * tanh(SL_lambda * ( 0.75 - Y/SL_L ));
                }
                uy = SL_delta * SL_U * sin(2*pi*(X/SL_L+0.25));
        } else {
                ux=0;
                uy=0;
        }
        ux = Velocity+ux;
         
        real_t uu[2];
        uu[0] = ux*rho;
        uu[1] = uy*rho;
        SetEquilibrum(
                rho,
                uu
        );
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
 real_t c[9],m[9];
  real_t u[2],usq,d;
  d = getRho();
  u[0] = (f[8] - f[7] - f[6] + f[5] - f[3] + f[1]);
  u[1] = (-f[8] - f[7] + f[6] + f[5] - f[4] + f[2]);
  usq = u[0]*u[0] + u[1]*u[1];
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
   real_t  w[5] = {1.0/(3*nu+0.5),1.,1.,1.,1.0};  // defining relaxation rate for first cummulants
//     real_t w[5] = {0.9,1.,1.,1.,1.};

for (int i = 0;i<9;i++) m[i] = f[i];
//First determing moments from density-probability function
  
  m[0] = m[3] + m[1] + m[0]; 
  m[1] = -m[3] + m[1];
  m[3] = m[1] + m[3]*2.; 
  m[2] = m[6] + m[5] + m[2];
  m[5] = -m[6] + m[5];
  m[6] = m[5] + m[6]*2.; 
  m[4] = m[7] + m[8] + m[4];
  m[8] = -m[7] + m[8];
  m[7] = m[8] + m[7]*2.; 
  m[0] = m[4] + m[2] + m[0];
  m[2] = -m[4] + m[2];
  m[4] = m[2] + m[4]*2.; 
  m[1] = m[8] + m[5] + m[1];
  m[5] = -m[8] + m[5];
  m[8] = m[5] + m[8]*2.; 
  m[3] = m[7] + m[6] + m[3];
  m[6] = -m[7] + m[6];
  m[7] = m[6] + m[7]*2.; 
  
//Cummulant calculation from moments
  c[0] = m[0];
  c[1] = m[1]/m[0];
  c[3] = ( -c[1]*m[1] + m[3] )/m[0];
  c[2] = m[2]/m[0];
  c[5] = ( -c[1]*m[2] + m[5] )/m[0];
  c[6] = ( -c[5]*m[1] - c[3]*m[2] - c[1]*m[5] + m[6] )/m[0];
  c[4] = ( -c[2]*m[2] + m[4] )/m[0];
  c[8] = ( -c[1]*m[4] + m[8] - c[5]*m[2]*2. )/m[0];
  c[7] = ( -c[8]*m[1] - c[3]*m[4] - c[1]*m[8] + m[7] + ( -c[6]*m[2] - c[5]*m[5] )*2. )/m[0];
//Cumulant relaxation:
 real_t  a = (c[3] + c[4]);
 real_t  b = (c[3] - c[4]);
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

  m[0] = m[0];
  m[1] = c[1]*m[0];
  m[3] = c[3]*m[0] + c[1]*m[1];
  m[2] = c[2]*m[0];
  m[5] = c[5]*m[0] + c[1]*m[2];
  m[6] = c[6]*m[0] + c[5]*m[1] + c[3]*m[2] + c[1]*m[5];
  m[4] = c[4]*m[0] + c[2]*m[2];
  m[8] = c[8]*m[0] + c[1]*m[4] + c[5]*m[2]*2.;
  m[7] = c[7]*m[0] + c[8]*m[1] + c[3]*m[4] + c[1]*m[8] + ( c[6]*m[2] + c[5]*m[5] )*2.;
 
 //Transformation from moment to density distribution function

  m[0] = -m[3] + m[0];
  m[1] = ( m[3] + m[1] )/2.;
  m[3] = m[3] - m[1]; 
  m[2] = -m[6] + m[2];
  m[5] = ( m[6] + m[5] )/2.;
  m[6] = m[6] - m[5]; 
  m[4] = -m[7] + m[4];
  m[8] = ( m[7] + m[8] )/2.;
  m[7] = m[7] - m[8]; 
  m[0] = -m[4] + m[0];
  m[2] = ( m[4] + m[2] )/2.;
  m[4] = m[4] - m[2]; 
  m[1] = -m[8] + m[1];
  m[5] = ( m[8] + m[5] )/2.;
  m[8] = m[8] - m[5]; 
  m[3] = -m[7] + m[3];
  m[6] = ( m[7] + m[6] )/2.;
  m[7] = m[7] - m[6]; 

 for (int i = 0;i<9;i++) f[i] = m[i];
}
/*CudaDeviceFunction void CollisionMRT()
{
 real_t c[9];//,m[9];
  real_t u[2],usq,d;
  d = getRho();
  u[0] = (f[8] - f[7] - f[6] + f[5] - f[3] + f[1]);
  u[1] = (-f[8] - f[7] + f[6] + f[5] - f[4] + f[2]);
  usq = u[0]*u[0] + u[1]*u[1];
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
   real_t  w[5] = {1.0/(3*nu+0.5),1.,1.,1.,1.0};  // defining relaxation rate for first cummulants


//for (int i = 0;i<9;i++) m[i] = f[i];
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
  
//Cummulant calculation from moments
  c[0] = f[0];
  c[1] = f[1]/f[0];
  c[3] = ( -c[1]*f[1] + f[3] )/f[0];
  c[2] = f[2]/f[0];
  c[5] = ( -c[1]*f[2] + f[5] )/f[0];
  c[6] = ( -c[5]*f[1] - c[3]*f[2] - c[1]*f[5] + f[6] )/f[0];
  c[4] = ( -c[2]*f[2] + f[4] )/f[0];
  c[8] = ( -c[1]*f[4] + f[8] - c[5]*f[2]*2. )/f[0];
  c[7] = ( -c[8]*f[1] - c[3]*f[4] - c[1]*f[8] + f[7] + ( -c[6]*f[2] - c[5]*f[5] )*2. )/f[0];
//Cummulant relaxation:

 c[1] = -c[1];
 c[3] = (1 - w[1])*c[3] + w[1];
 c[2] =-c[2];
 c[4] = (1 - w[1])*c[4] + w[1];
 c[5] =  (1- w[0])*c[5];
 c[6] =  (1 - w[2])*c[6];
 c[7] =  (1 - w[3])*c[7];
 c[8] = (1 - w[2])*c[8]; 


// moment calculation from cummulants

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
}*/ 
