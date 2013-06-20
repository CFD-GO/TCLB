
#define S2 1.3333
#define S3 1.0
#define S5 1.0
#define S7 1.0
#define S8 omega
#define S9 omega



CudaDeviceFunction type_f getRho(){
	return    f[ 0 ] + f[ 1 ] + f[ 2 ] + f[ 3 ] + f[ 4 ] + f[ 5 ] + f[ 6 ] + f[ 7 ] + f[ 8 ] ;
}
    
CudaDeviceFunction type_v getU(){
	type_f d =    f[ 0 ] + f[ 1 ] + f[ 2 ] + f[ 3 ] + f[ 4 ] + f[ 5 ] + f[ 6 ] + f[ 7 ] + f[ 8 ];
	type_v u;
u.x =    f[ 1 ] - f[ 3 ] + f[ 5 ] - f[ 6 ] - f[ 7 ] + f[ 8 ] ;
u.y =    f[ 2 ] - f[ 4 ] + f[ 5 ] + f[ 6 ] - f[ 7 ] - f[ 8 ] ;

	u.x /= d;
	u.y /= d;
	u.z = 0.0;
	return u;
}

CudaDeviceFunction float2 Color() {
        float2 ret;
        type_v u = getU();
        ret.x = sqrt(u.x*u.x + u.y*u.y);
        if (NodeType == NODE_Solid){
                ret.y = 0;
        } else {
                ret.y = 1;
        }
        return ret;
}


CudaDeviceFunction void BounceBack()
{
     type_f uf;
     uf =    f[ 3 ] ;
     f[ 3 ] =    f[ 1 ] ;
     f[ 1 ] =    uf ;
     uf =    f[ 4 ] ;
     f[ 4 ] =    f[ 2 ] ;
     f[ 2 ] =    uf ;
     uf =    f[ 7 ] ;
     f[ 7 ] =    f[ 5 ] ;
     f[ 5 ] =    uf ;
     uf =    f[ 8 ] ;
     f[ 8 ] =    f[ 6 ] ;
     f[ 6 ] =    uf ;
}

CudaDeviceFunction void WVelocity()
{
        type_f rho, ru;
	const type_f u[2] = {UX,0.};
	rho = ( f[0] + f[2] + f[4] + 2.*(f[3] + f[7] + f[6]) ) / (1. - u[0]);
	ru = rho * u[0];
	f[1] = f[3] + (2./3.) * ru;
	f[5] = f[7] + (1./6.) * ru + (1./2.)*(f[4] - f[2]);
	f[8] = f[6] + (1./6.) * ru + (1./2.)*(f[2] - f[4]);
}

CudaDeviceFunction void EPressure()
{
        type_f ru, ux0;
	const type_f rho = 1.0;
	ux0 = -1. + ( f[0] + f[2] + f[4] + 2.*(f[1] + f[5] + f[8]) ) / rho;
	ru = rho * ux0;
	f[3] = f[1] - (2./3.) * ru;
	f[7] = f[5] - (1./6.) * ru + (1./2.)*(f[2] - f[4]);
	f[6] = f[8] - (1./6.) * ru + (1./2.)*(f[4] - f[2]);
}


CudaDeviceFunction void Run() {
    switch (NodeType & NODE_BOUNDARY) {
	case NODE_Solid:
	case NODE_Wall:
		BounceBack();
		break;
	case NODE_WVelocity:
		WVelocity();
		break;
	case NODE_WPressure:
		WVelocity();
		break;
	case NODE_EPressure:
		EPressure();
		break;
    }
    if (NodeType & NODE_COLLISION)
    {
		CollisionMRT();
    }
}

CudaDeviceFunction void SetEquilibrum(const type_f d, const type_f u[2])
{	type_f usq, uf;
usq =    3.0000000000e+00*(u[0]*u[0]) + 3.0000000000e+00*(u[1]*u[1]) ;

//-- 1 -------------------------------------------------
uf =    0 ;
uf =    1 + uf + 5.0000000000e-01*(uf*uf) - 5.0000000000e-01*usq ;
uf =    d*uf ;
f[ 0 ] =    4.4444444444e-01*uf ;
//-- 2 -------------------------------------------------
uf =    3.0000000000e+00*u[0] ;
uf =    1 + uf + 5.0000000000e-01*(uf*uf) - 5.0000000000e-01*usq ;
uf =    d*uf ;
f[ 1 ] =    1.1111111111e-01*uf ;
//-- 3 -------------------------------------------------
uf =    3.0000000000e+00*u[1] ;
uf =    1 + uf + 5.0000000000e-01*(uf*uf) - 5.0000000000e-01*usq ;
uf =    d*uf ;
f[ 2 ] =    1.1111111111e-01*uf ;
//-- 4 -------------------------------------------------
uf =  - 3.0000000000e+00*u[0] ;
uf =    1 + uf + 5.0000000000e-01*(uf*uf) - 5.0000000000e-01*usq ;
uf =    d*uf ;
f[ 3 ] =    1.1111111111e-01*uf ;
//-- 5 -------------------------------------------------
uf =  - 3.0000000000e+00*u[1] ;
uf =    1 + uf + 5.0000000000e-01*(uf*uf) - 5.0000000000e-01*usq ;
uf =    d*uf ;
f[ 4 ] =    1.1111111111e-01*uf ;
//-- 6 -------------------------------------------------
uf =    3.0000000000e+00*u[0] + 3.0000000000e+00*u[1] ;
uf =    1 + uf + 5.0000000000e-01*(uf*uf) - 5.0000000000e-01*usq ;
uf =    d*uf ;
f[ 5 ] =    2.7777777778e-02*uf ;
//-- 7 -------------------------------------------------
uf =  - 3.0000000000e+00*u[0] + 3.0000000000e+00*u[1] ;
uf =    1 + uf + 5.0000000000e-01*(uf*uf) - 5.0000000000e-01*usq ;
uf =    d*uf ;
f[ 6 ] =    2.7777777778e-02*uf ;
//-- 8 -------------------------------------------------
uf =  - 3.0000000000e+00*u[0] - 3.0000000000e+00*u[1] ;
uf =    1 + uf + 5.0000000000e-01*(uf*uf) - 5.0000000000e-01*usq ;
uf =    d*uf ;
f[ 7 ] =    2.7777777778e-02*uf ;
//-- 9 -------------------------------------------------
uf =    3.0000000000e+00*u[0] - 3.0000000000e+00*u[1] ;
uf =    1 + uf + 5.0000000000e-01*(uf*uf) - 5.0000000000e-01*usq ;
uf =    d*uf ;
f[ 8 ] =    2.7777777778e-02*uf ;


}

CudaDeviceFunction void Init() {
	type_f u[2] = {UX,0.};
	type_f d = 1.0;
	SetEquilibrum(d,u);
}


CudaDeviceFunction void CollisionMRT()
{
	type_f u[2], usq, d, R[6], uf;
d =    f[ 0 ] + f[ 1 ] + f[ 2 ] + f[ 3 ] + f[ 4 ] + f[ 5 ] + f[ 6 ] + f[ 7 ] + f[ 8 ] ;
u[0] =    f[ 1 ] - f[ 3 ] + f[ 5 ] - f[ 6 ] - f[ 7 ] + f[ 8 ] ;
u[1] =    f[ 2 ] - f[ 4 ] + f[ 5 ] + f[ 6 ] - f[ 7 ] - f[ 8 ] ;
R[0] =  - 4*f[ 0 ] - f[ 1 ] - f[ 2 ] - f[ 3 ] - f[ 4 ] + 2*f[ 5 ] + 2*f[ 6 ] + 2*f[ 7 ] + 2*f[ 8 ] ;
R[1] =    4*f[ 0 ] - 2*f[ 1 ] - 2*f[ 2 ] - 2*f[ 3 ] - 2*f[ 4 ] + f[ 5 ] + f[ 6 ] + f[ 7 ] + f[ 8 ] ;
R[2] =  - 2*f[ 1 ] + 2*f[ 3 ] + f[ 5 ] - f[ 6 ] - f[ 7 ] + f[ 8 ] ;
R[3] =  - 2*f[ 2 ] + 2*f[ 4 ] + f[ 5 ] + f[ 6 ] - f[ 7 ] - f[ 8 ] ;
R[4] =    f[ 1 ] - f[ 2 ] + f[ 3 ] - f[ 4 ] ;
R[5] =    f[ 5 ] - f[ 6 ] + f[ 7 ] - f[ 8 ] ;
usq =    (u[0]*u[0]) + (u[1]*u[1]) ;


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
f[ 0 ] =    d - 4*R[0] + 4*R[1] ;
f[ 1 ] =    d + u[0] - R[0] - 2*R[1] - 2*R[2] + R[4] ;
f[ 2 ] =    d + u[1] - R[0] - 2*R[1] - 2*R[3] - R[4] ;
f[ 3 ] =    d - u[0] - R[0] - 2*R[1] + 2*R[2] + R[4] ;
f[ 4 ] =    d - u[1] - R[0] - 2*R[1] + 2*R[3] - R[4] ;
f[ 5 ] =    d + u[0] + u[1] + 2*R[0] + R[1] + R[2] + R[3] + R[5] ;
f[ 6 ] =    d - u[0] + u[1] + 2*R[0] + R[1] - R[2] + R[3] - R[5] ;
f[ 7 ] =    d - u[0] - u[1] + 2*R[0] + R[1] - R[2] - R[3] + R[5] ;
f[ 8 ] =    d + u[0] - u[1] + 2*R[0] + R[1] + R[2] - R[3] - R[5] ;

}


