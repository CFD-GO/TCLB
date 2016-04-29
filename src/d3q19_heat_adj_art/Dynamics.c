

CudaDeviceFunction real_t getRho(){
	return    f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9 + f10 + f11 + f12 + f13 + f14 + f15 + f16 + f17 + f18;
}

CudaDeviceFunction real_t getW(){
	return w;
}

CudaDeviceFunction real_t getT(){
	return (   T0 + T1 + T2 + T3 + T4 + T5 + T6)/(   f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9 + f10 + f11 + f12 + f13 + f14 + f15 + f16 + f17 + f18);
}
    
CudaDeviceFunction vector_t getU(){
	real_t d = getRho();
	vector_t u;
u.x =    f1 - f2 + f7 - f8 + f9 - f10 + f11 - f12 + f13 - f14 ;
u.y =    f3 - f4 + f7 + f8 - f9 - f10 + f15 - f16 + f17 - f18 ;
u.z =    f5 - f6 + f11 + f12 - f13 - f14 + f15 + f16 - f17 - f18 ;

	u.x /= d;
	u.y /= d;
	u.z /= d;
	return u;
}

CudaDeviceFunction float2 Color() {
        float2 ret;
//        vector_t u = getU();
//        ret.x = sqrt(u.x*u.x + u.y*u.y + u.z*u.z);
        ret.x = getT();
        ret.y = w;
        return ret;
}

CudaDeviceFunction void Collision()
{

}


CudaDeviceFunction void BounceBack()
{
     real_t uf;
uf =    f10 ;
f10 =    f7 ;
f7 =    uf ;
uf =    f14 ;
f14 =    f11 ;
f11 =    uf ;
uf =    f2 ;
f2 =    f1 ;
f1 =    uf ;
uf =    f18 ;
f18 =    f15 ;
f15 =    uf ;
uf =    f4 ;
f4 =    f3 ;
f3 =    uf ;
uf =    f6 ;
f6 =    f5 ;
f5 =    uf ;
uf =    f17 ;
f17 =    f16 ;
f16 =    uf ;
uf =    f9 ;
f9 =    f8 ;
f8 =    uf ;
uf =    f13 ;
f13 =    f12 ;
f12 =    uf ;
uf =    T2 ;
T2 =    T1 ;
T1 =    uf ;
uf =    T4 ;
T4 =    T3 ;
T3 =    uf ;
uf =    T6 ;
T6 =    T5 ;
T5 =    uf ;
}

CudaDeviceFunction void EVelocity()
{

}

CudaDeviceFunction void eqWVelocity()
{
}






CudaDeviceFunction void WVelocity()
{
     real_t rho, Nxy, Nxz;
	real_t ux = Velocity;

rho =    f2 + f8 + f10 + f12 + f14 ;
rho =    f0 + f3 + f4 + f5 + f6 + f15 + f16 + f17 + f18 + 2*rho ;

	rho = rho / (1. - ux);


	Nxy = (   f3 + f15 + f17 - f4 - f16 - f18)/2.;
	Nxz = (   f5 + f15 + f16 - f6 - f17 - f18)/2.;

	f1 = f2 + rho * ux / 3.;
	f9 = f8 + rho * ux / 6. + Nxy;
	f7 = f10 + rho * ux / 6. - Nxy;
	f11 = f14 + rho * ux / 6. - Nxz;
	f13 = f12 + rho * ux / 6. + Nxz;

        rho = Temperature;
        T0 =    2.5000000000e-01*rho ;
T1 =    1.2500000000e-01*rho + 5.0000000000e-01*rho*ux ;
T2 =    1.2500000000e-01*rho - 5.0000000000e-01*rho*ux ;
T3 =    1.2500000000e-01*rho ;
T4 =    1.2500000000e-01*rho ;
T5 =    1.2500000000e-01*rho ;
T6 =    1.2500000000e-01*rho ;


}

CudaDeviceFunction void WPressure()
{
     real_t rho, Nxy, Nxz;
	real_t ux;
	rho = 1.0+Pressure*3.0;
ux =    f2 + f8 + f10 + f12 + f14 ;
ux =    f0 + f3 + f4 + f5 + f6 + f15 + f16 + f17 + f18 + 2*ux ;

	ux = 1. - ux / rho;

	Nxy = (   f3 + f15 + f17 - f4 - f16 - f18)/2.;
	Nxz = (   f5 + f15 + f16 - f6 - f17 - f18)/2.;

	f1 = f2 + rho * ux / 3.;
	f9 = f8 + rho * ux / 6. + Nxy;
	f7 = f10 + rho * ux / 6. - Nxy;
	f11 = f14 + rho * ux / 6. - Nxz;
	f13 = f12 + rho * ux / 6. + Nxz;

        rho = Temperature;
        T0 =    2.5000000000e-01*rho ;
T1 =    1.2500000000e-01*rho + 5.0000000000e-01*rho*ux ;
T2 =    1.2500000000e-01*rho - 5.0000000000e-01*rho*ux ;
T3 =    1.2500000000e-01*rho ;
T4 =    1.2500000000e-01*rho ;
T5 =    1.2500000000e-01*rho ;
T6 =    1.2500000000e-01*rho ;


}

CudaDeviceFunction void WPressureLimited()
{
     real_t rho, Nxy, Nxz, SF, ux;
SF =    f2 + f8 + f10 + f12 + f14 ;
SF =    f0 + f3 + f4 + f5 + f6 + f15 + f16 + f17 + f18 + 2*SF ;

	rho = 1.0+Pressure*3.0;
	ux = 1. - SF / rho;
	if (ux > Velocity) {
		ux = Velocity;
		rho = SF / (1. - ux);
	}	

	Nxy = (   f3 + f15 + f17 - f4 - f16 - f18)/2.;
	Nxz = (   f5 + f15 + f16 - f6 - f17 - f18)/2.;

	f1 = f2 + rho * ux / 3.;
	f9 = f8 + rho * ux / 6. + Nxy;
	f7 = f10 + rho * ux / 6. - Nxy;
	f11 = f14 + rho * ux / 6. - Nxz;
	f13 = f12 + rho * ux / 6. + Nxz;

        rho = Temperature;
        T0 =    2.5000000000e-01*rho ;
T1 =    1.2500000000e-01*rho + 5.0000000000e-01*rho*ux ;
T2 =    1.2500000000e-01*rho - 5.0000000000e-01*rho*ux ;
T3 =    1.2500000000e-01*rho ;
T4 =    1.2500000000e-01*rho ;
T5 =    1.2500000000e-01*rho ;
T6 =    1.2500000000e-01*rho ;


}



CudaDeviceFunction void EPressure()
{
     real_t rho = 1.0;
     real_t Nxy, Nxz;
     real_t ux;
ux =    f1 + f7 + f9 + f11 + f13 ;
ux =    f0 + f3 + f4 + f5 + f6 + f15 + f16 + f17 + f18 + 2*ux ;

	ux =  -1. + ux / rho;

	Nxy = (   f3 + f15 + f17 - f4 - f16 - f18)/2;
	Nxz = (   f5 + f15 + f16 - f6 - f17 - f18)/2;

	f2 = f1 - rho * ux / 3.0;
	f8 = f9 - rho * ux / 6.0 - Nxy;
	f10 = f7 - rho * ux / 6.0 + Nxy;
	f14 = f11 - rho * ux / 6.0 + Nxz;
	f12 = f13 - rho * ux / 6.0 - Nxz;

	
        rho =2*T1 +    T0 + T3 + T4 + T5 + T6;
	rho = rho/(1.+ux);
	T2 = T1 - rho*ux;
}
// rho = sum T = 2T1+Trest - rho*ux
// rho(1+ux) = 2T1 + Trest

CudaDeviceFunction void Run() {
//	printf("Run %d %d -> (%d,%d)\n", CudaBlock.x, CudaBlock.y, X, Y);
    switch (NodeType & NODE_BOUNDARY) {
	case NODE_WPressureL:
		WPressureLimited();
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
	case NODE_Wall:
		BounceBack();
                break;
    }
    switch (NodeType & NODE_COLLISION) {
	case NODE_BGK:
		Collision();
		break;
	case NODE_MRT:
		CollisionMRT();
		break;
    }
    if (NodeType & NODE_DESIGNSPACE) {
            AddToMaterialPenalty(w*(1-w));
    }
}

CudaDeviceFunction void SetEquilibrum(real_t rho, real_t Jx, real_t Jy, real_t Jz, real_t T)
{
	f0 =    3.3333333333e-01*rho - 5.0000000000e-01*(Jx*Jx) - 5.0000000000e-01*(Jy*Jy) - 5.0000000000e-01*(Jz*Jz) ;
f1 =    5.5555555556e-02*rho + 1.6666666667e-01*Jx + 1.6666666667e-01*(Jx*Jx) - 8.3333333333e-02*(Jy*Jy) - 8.3333333333e-02*(Jz*Jz) ;
f2 =    5.5555555556e-02*rho - 1.6666666667e-01*Jx + 1.6666666667e-01*(Jx*Jx) - 8.3333333333e-02*(Jy*Jy) - 8.3333333333e-02*(Jz*Jz) ;
f3 =    5.5555555556e-02*rho - 8.3333333333e-02*(Jx*Jx) + 1.6666666667e-01*Jy + 1.6666666667e-01*(Jy*Jy) - 8.3333333333e-02*(Jz*Jz) ;
f4 =    5.5555555556e-02*rho - 8.3333333333e-02*(Jx*Jx) - 1.6666666667e-01*Jy + 1.6666666667e-01*(Jy*Jy) - 8.3333333333e-02*(Jz*Jz) ;
f5 =    5.5555555556e-02*rho - 8.3333333333e-02*(Jx*Jx) - 8.3333333333e-02*(Jy*Jy) + 1.6666666667e-01*Jz + 1.6666666667e-01*(Jz*Jz) ;
f6 =    5.5555555556e-02*rho - 8.3333333333e-02*(Jx*Jx) - 8.3333333333e-02*(Jy*Jy) - 1.6666666667e-01*Jz + 1.6666666667e-01*(Jz*Jz) ;
f7 =    2.7777777778e-02*rho + 8.3333333333e-02*Jx + 8.3333333333e-02*(Jx*Jx) + 8.3333333333e-02*Jy + 2.5000000000e-01*Jx*Jy + 8.3333333333e-02*(Jy*Jy) - 4.1666666667e-02*(Jz*Jz) ;
f8 =    2.7777777778e-02*rho - 8.3333333333e-02*Jx + 8.3333333333e-02*(Jx*Jx) + 8.3333333333e-02*Jy - 2.5000000000e-01*Jx*Jy + 8.3333333333e-02*(Jy*Jy) - 4.1666666667e-02*(Jz*Jz) ;
f9 =    2.7777777778e-02*rho + 8.3333333333e-02*Jx + 8.3333333333e-02*(Jx*Jx) - 8.3333333333e-02*Jy - 2.5000000000e-01*Jx*Jy + 8.3333333333e-02*(Jy*Jy) - 4.1666666667e-02*(Jz*Jz) ;
f10 =    2.7777777778e-02*rho - 8.3333333333e-02*Jx + 8.3333333333e-02*(Jx*Jx) - 8.3333333333e-02*Jy + 2.5000000000e-01*Jx*Jy + 8.3333333333e-02*(Jy*Jy) - 4.1666666667e-02*(Jz*Jz) ;
f11 =    2.7777777778e-02*rho + 8.3333333333e-02*Jx + 8.3333333333e-02*(Jx*Jx) - 4.1666666667e-02*(Jy*Jy) + 8.3333333333e-02*Jz + 2.5000000000e-01*Jx*Jz + 8.3333333333e-02*(Jz*Jz) ;
f12 =    2.7777777778e-02*rho - 8.3333333333e-02*Jx + 8.3333333333e-02*(Jx*Jx) - 4.1666666667e-02*(Jy*Jy) + 8.3333333333e-02*Jz - 2.5000000000e-01*Jx*Jz + 8.3333333333e-02*(Jz*Jz) ;
f13 =    2.7777777778e-02*rho + 8.3333333333e-02*Jx + 8.3333333333e-02*(Jx*Jx) - 4.1666666667e-02*(Jy*Jy) - 8.3333333333e-02*Jz - 2.5000000000e-01*Jx*Jz + 8.3333333333e-02*(Jz*Jz) ;
f14 =    2.7777777778e-02*rho - 8.3333333333e-02*Jx + 8.3333333333e-02*(Jx*Jx) - 4.1666666667e-02*(Jy*Jy) - 8.3333333333e-02*Jz + 2.5000000000e-01*Jx*Jz + 8.3333333333e-02*(Jz*Jz) ;
f15 =    2.7777777778e-02*rho - 4.1666666667e-02*(Jx*Jx) + 8.3333333333e-02*Jy + 8.3333333333e-02*(Jy*Jy) + 8.3333333333e-02*Jz + 2.5000000000e-01*Jy*Jz + 8.3333333333e-02*(Jz*Jz) ;
f16 =    2.7777777778e-02*rho - 4.1666666667e-02*(Jx*Jx) - 8.3333333333e-02*Jy + 8.3333333333e-02*(Jy*Jy) + 8.3333333333e-02*Jz - 2.5000000000e-01*Jy*Jz + 8.3333333333e-02*(Jz*Jz) ;
f17 =    2.7777777778e-02*rho - 4.1666666667e-02*(Jx*Jx) + 8.3333333333e-02*Jy + 8.3333333333e-02*(Jy*Jy) - 8.3333333333e-02*Jz - 2.5000000000e-01*Jy*Jz + 8.3333333333e-02*(Jz*Jz) ;
f18 =    2.7777777778e-02*rho - 4.1666666667e-02*(Jx*Jx) - 8.3333333333e-02*Jy + 8.3333333333e-02*(Jy*Jy) - 8.3333333333e-02*Jz + 2.5000000000e-01*Jy*Jz + 8.3333333333e-02*(Jz*Jz) ;

	Jx /= rho;
	Jy /= rho;
	Jz /= rho;
	T0 =    2.5000000000e-01*T ;
T1 =    1.2500000000e-01*T + 5.0000000000e-01*T*Jx ;
T2 =    1.2500000000e-01*T - 5.0000000000e-01*T*Jx ;
T3 =    1.2500000000e-01*T + 5.0000000000e-01*T*Jy ;
T4 =    1.2500000000e-01*T - 5.0000000000e-01*T*Jy ;
T5 =    1.2500000000e-01*T + 5.0000000000e-01*T*Jz ;
T6 =    1.2500000000e-01*T - 5.0000000000e-01*T*Jz ;

	w = 1;
}

CudaDeviceFunction void Init() {
	SetEquilibrum(1.0, Velocity, 0., 0.,Temperature);
	if ((NodeType & NODE_BOUNDARY) == NODE_Solid) {
//		w = SolidLevel;
  w=0.01;
	}
}

CudaDeviceFunction void CollisionMRT()
{
	real_t Jx,Jy,Jz, rho, omT, T;
	real_t    R0,   R1,   R2,   R3,   R4,   R5,   R6,   R7,   R8,   R9,   R10,   R11,   R12,   R13,   R14;
real_t omega = 1.0/(0.5+3.*nu);
	#define S1  0.0
	#define S2  1.0//19
	#define S3  1.0//4
	#define S4  0.0
	#define S5  1.0//2
	#define S6  0.0
	#define S7  1.0//2
	#define S8  0.0
	#define S9  1.0//2
	#define S10 omega
	#define S11 1.0//4
	#define S12 omega
	#define S13 1.0//4
	#define S14 omega
	#define S15 omega
	#define S16 omega
	#define S17 1.0//98
	#define S18 1.0//98
	#define S19 1.0//98
rho =    f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9 + f10 + f11 + f12 + f13 + f14 + f15 + f16 + f17 + f18 ;
R0 =  - 30*f0 - 11*f1 - 11*f2 - 11*f3 - 11*f4 - 11*f5 - 11*f6 + 8*f7 + 8*f8 + 8*f9 + 8*f10 + 8*f11 + 8*f12 + 8*f13 + 8*f14 + 8*f15 + 8*f16 + 8*f17 + 8*f18 ;
R1 =    12*f0 - 4*f1 - 4*f2 - 4*f3 - 4*f4 - 4*f5 - 4*f6 + f7 + f8 + f9 + f10 + f11 + f12 + f13 + f14 + f15 + f16 + f17 + f18 ;
Jx =    f1 - f2 + f7 - f8 + f9 - f10 + f11 - f12 + f13 - f14 ;
R2 =  - 4*f1 + 4*f2 + f7 - f8 + f9 - f10 + f11 - f12 + f13 - f14 ;
Jy =    f3 - f4 + f7 + f8 - f9 - f10 + f15 - f16 + f17 - f18 ;
R3 =  - 4*f3 + 4*f4 + f7 + f8 - f9 - f10 + f15 - f16 + f17 - f18 ;
Jz =    f5 - f6 + f11 + f12 - f13 - f14 + f15 + f16 - f17 - f18 ;
R4 =  - 4*f5 + 4*f6 + f11 + f12 - f13 - f14 + f15 + f16 - f17 - f18 ;
R5 =    2*f1 + 2*f2 - f3 - f4 - f5 - f6 + f7 + f8 + f9 + f10 + f11 + f12 + f13 + f14 - 2*f15 - 2*f16 - 2*f17 - 2*f18 ;
R6 =  - 4*f1 - 4*f2 + 2*f3 + 2*f4 + 2*f5 + 2*f6 + f7 + f8 + f9 + f10 + f11 + f12 + f13 + f14 - 2*f15 - 2*f16 - 2*f17 - 2*f18 ;
R7 =    f3 + f4 - f5 - f6 + f7 + f8 + f9 + f10 - f11 - f12 - f13 - f14 ;
R8 =  - 2*f3 - 2*f4 + 2*f5 + 2*f6 + f7 + f8 + f9 + f10 - f11 - f12 - f13 - f14 ;
R9 =    f7 - f8 - f9 + f10 ;
R10 =    f15 - f16 - f17 + f18 ;
R11 =    f11 - f12 - f13 + f14 ;
R12 =    f7 - f8 + f9 - f10 - f11 + f12 - f13 + f14 ;
R13 =  - f7 - f8 + f9 + f10 + f15 - f16 + f17 - f18 ;
R14 =    f11 + f12 - f13 - f14 - f15 - f16 + f17 + f18 ;
	   R0 = (1.-   S2)*(   R0 + 1.1000000000e+01*rho - 1.9000000000e+01*(Jx*Jx) - 1.9000000000e+01*(Jy*Jy) - 1.9000000000e+01*(Jz*Jz) );
	   R1 = (1.-   S3)*(   R1 - 3.0000000000e+00*rho + 5.5000000000e+00*(Jx*Jx) + 5.5000000000e+00*(Jy*Jy) + 5.5000000000e+00*(Jz*Jz) );
	   R2 = (1.-   S5)*(   R2 + 6.6666666667e-01*Jx );
	   R3 = (1.-   S7)*(   R3 + 6.6666666667e-01*Jy );
	   R4 = (1.-   S9)*(   R4 + 6.6666666667e-01*Jz );
	   R5 = (1.-   S10)*(   R5 - 2.0000000000e+00*(Jx*Jx) + (Jy*Jy) + (Jz*Jz) );
	   R6 = (1.-   S11)*(   R6 + (Jx*Jx) - 5.0000000000e-01*(Jy*Jy) - 5.0000000000e-01*(Jz*Jz) );
	   R7 = (1.-   S12)*(   R7 - (Jy*Jy) + (Jz*Jz) );
	   R8 = (1.-   S13)*(   R8 + 5.0000000000e-01*(Jy*Jy) - 5.0000000000e-01*(Jz*Jz) );
	   R9 = (1.-   S14)*(   R9 - Jx*Jy );
	   R10 = (1.-   S15)*(   R10 - Jy*Jz );
	   R11 = (1.-   S16)*(   R11 - Jx*Jz );
	   R12 = (1.-   S17)*(   R12 );
	   R13 = (1.-   S18)*(   R13 );
	   R14 = (1.-   S19)*(   R14 );

//	omT = sqrt(Jx*Jx+Jy*Jy+Jz*Jz);
//	omT = exp(log(w)*omT);
	omT = w*2-1;
        Jx *= omT;
        Jy *= omT;  
        Jz *= omT;
	   R0 +=  - 1.1000000000e+01*rho + 1.9000000000e+01*(Jx*Jx) + 1.9000000000e+01*(Jy*Jy) + 1.9000000000e+01*(Jz*Jz);
	   R1 +=    3.0000000000e+00*rho - 5.5000000000e+00*(Jx*Jx) - 5.5000000000e+00*(Jy*Jy) - 5.5000000000e+00*(Jz*Jz);
	   R2 +=  - 6.6666666667e-01*Jx;
	   R3 +=  - 6.6666666667e-01*Jy;
	   R4 +=  - 6.6666666667e-01*Jz;
	   R5 +=    2.0000000000e+00*(Jx*Jx) - (Jy*Jy) - (Jz*Jz);
	   R6 +=  - (Jx*Jx) + 5.0000000000e-01*(Jy*Jy) + 5.0000000000e-01*(Jz*Jz);
	   R7 +=    (Jy*Jy) - (Jz*Jz);
	   R8 +=  - 5.0000000000e-01*(Jy*Jy) + 5.0000000000e-01*(Jz*Jz);
	   R9 +=    Jx*Jy;
	   R10 +=    Jy*Jz;
	   R11 +=    Jx*Jz;
	   R12 +=    0;
	   R13 +=    0;
	   R14 +=    0;
	   rho /= 19.000000;
	   R0 /= 2394.000000;
	   R1 /= 252.000000;
	   Jx /= 10.000000;
	   R2 /= 40.000000;
	   Jy /= 10.000000;
	   R3 /= 40.000000;
	   Jz /= 10.000000;
	   R4 /= 40.000000;
	   R5 /= 36.000000;
	   R6 /= 72.000000;
	   R7 /= 12.000000;
	   R8 /= 24.000000;
	   R9 /= 4.000000;
	   R10 /= 4.000000;
	   R11 /= 4.000000;
	   R12 /= 8.000000;
	   R13 /= 8.000000;
	   R14 /= 8.000000;
f0 =    rho - 30*R0 + 12*R1 ;
f1 =    rho - 11*R0 - 4*R1 + Jx - 4*R2 + 2*R5 - 4*R6 ;
f2 =    rho - 11*R0 - 4*R1 - Jx + 4*R2 + 2*R5 - 4*R6 ;
f3 =    rho - 11*R0 - 4*R1 + Jy - 4*R3 - R5 + 2*R6 + R7 - 2*R8 ;
f4 =    rho - 11*R0 - 4*R1 - Jy + 4*R3 - R5 + 2*R6 + R7 - 2*R8 ;
f5 =    rho - 11*R0 - 4*R1 + Jz - 4*R4 - R5 + 2*R6 - R7 + 2*R8 ;
f6 =    rho - 11*R0 - 4*R1 - Jz + 4*R4 - R5 + 2*R6 - R7 + 2*R8 ;
f7 =    rho + 8*R0 + R1 + Jx + R2 + Jy + R3 + R5 + R6 + R7 + R8 + R9 + R12 - R13 ;
f8 =    rho + 8*R0 + R1 - Jx - R2 + Jy + R3 + R5 + R6 + R7 + R8 - R9 - R12 - R13 ;
f9 =    rho + 8*R0 + R1 + Jx + R2 - Jy - R3 + R5 + R6 + R7 + R8 - R9 + R12 + R13 ;
f10 =    rho + 8*R0 + R1 - Jx - R2 - Jy - R3 + R5 + R6 + R7 + R8 + R9 - R12 + R13 ;
f11 =    rho + 8*R0 + R1 + Jx + R2 + Jz + R4 + R5 + R6 - R7 - R8 + R11 - R12 + R14 ;
f12 =    rho + 8*R0 + R1 - Jx - R2 + Jz + R4 + R5 + R6 - R7 - R8 - R11 + R12 + R14 ;
f13 =    rho + 8*R0 + R1 + Jx + R2 - Jz - R4 + R5 + R6 - R7 - R8 - R11 - R12 - R14 ;
f14 =    rho + 8*R0 + R1 - Jx - R2 - Jz - R4 + R5 + R6 - R7 - R8 + R11 + R12 - R14 ;
f15 =    rho + 8*R0 + R1 + Jy + R3 + Jz + R4 - 2*R5 - 2*R6 + R10 + R13 - R14 ;
f16 =    rho + 8*R0 + R1 - Jy - R3 + Jz + R4 - 2*R5 - 2*R6 - R10 - R13 - R14 ;
f17 =    rho + 8*R0 + R1 + Jy + R3 - Jz - R4 - 2*R5 - 2*R6 - R10 + R13 + R14 ;
f18 =    rho + 8*R0 + R1 - Jy - R3 - Jz - R4 - 2*R5 - 2*R6 + R10 - R13 + R14 ;


        Jx /= rho * 1.900000;
        Jy /= rho * 1.900000;
        Jz /= rho * 1.900000;

	omT = w*FluidAlpha + (1-w)*SolidAlpha;
	omT = 1/(0.5+omT*4);
T =    T0 + T1 + T2 + T3 + T4 + T5 + T6 ;
R1 =    T1 - T2 ;
R2 =    T3 - T4 ;
R3 =    T5 - T6 ;
R4 =    T1 + T2 ;
R5 =    T3 + T4 ;
R6 =    T5 + T6 ;
	   R1 = (1 - omT)*(   R1 - T*Jx );
	   R2 = (1 - omT)*(   R2 - T*Jy );
	   R3 = (1 - omT)*(   R3 - T*Jz );
	   R4 = (1 - omT)*(   R4 - 2.5000000000e-01*T );
	   R5 = (1 - omT)*(   R5 - 2.5000000000e-01*T );
	   R6 = (1 - omT)*(   R6 - 2.5000000000e-01*T );

	if ((NodeType & NODE_ADDITIONALS) == NODE_Heater) {
		T = Temperature;
	}
//	if ((NodeType & NODE_ADDITIONALS) == NODE_HeatSource) {
//		T += HeatSource;
//	}
//	if ((NodeType & NODE_OBJECTIVE) == NODE_Outlet) AddToObjective( T*Jx);
	if ((NodeType & NODE_OBJECTIVE) == NODE_Outlet) {
//		if ((T > -1.) && (T < 1.)) AddToObjective( (1-T*T)*Jx);
		AddToFlux( Jx );
		AddToHeatFlux( T*Jx );
		AddToHeatSquareFlux( T*T*Jx );
	}
	if ((NodeType & NODE_OBJECTIVE) == NODE_Thermometer) {
		AddToTemperatureAtPoint(T);
		if (T > LimitTemperature) {
			AddToHighTemperature((T-LimitTemperature)*(T-LimitTemperature));
		} else {
			AddToLowTemperature( (T-LimitTemperature)*(T-LimitTemperature));
		}
	}
//	if ((NodeType & NODE_OBJECTIVE) == NODE_Inlet) AddToObjective(-T*T);
	   R1 +=    T*Jx;
	   R2 +=    T*Jy;
	   R3 +=    T*Jz;
	   R4 +=    2.5000000000e-01*T;
	   R5 +=    2.5000000000e-01*T;
	   R6 +=    2.5000000000e-01*T;
T0 =    T - R4 - R5 - R6 ;
T1 =    5.0000000000e-01*R1 + 5.0000000000e-01*R4 ;
T2 =  - 5.0000000000e-01*R1 + 5.0000000000e-01*R4 ;
T3 =    5.0000000000e-01*R2 + 5.0000000000e-01*R5 ;
T4 =  - 5.0000000000e-01*R2 + 5.0000000000e-01*R5 ;
T5 =    5.0000000000e-01*R3 + 5.0000000000e-01*R6 ;
T6 =  - 5.0000000000e-01*R3 + 5.0000000000e-01*R6 ;

}
