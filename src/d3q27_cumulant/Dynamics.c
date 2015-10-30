/*-------------------------------------------------------------*/
/*  CLB - Cudne LB - Stencil Version                           */
/*     CUDA based Adjoint Lattice Boltzmann Solver             */
/*     Author: Lukasz Laniewski-Wollk                          */
/*     Developed at: Warsaw University of Technology - 2012    */
/*-------------------------------------------------------------*/


CudaDeviceFunction real_t getRho(){
	return f26 + f25 + f24 + f23 + f22 + f21 + f20 + f19 + f18 + f17 + f16 + f15 + f14 + f13 + f12 + f11 + f10 + f9 + f8 + f7 + f6 + f5 + f4 + f3 + f2 + f1 + f0;
}

CudaDeviceFunction real_t getP(){
	return ((f26 + f25 + f24 + f23 + f22 + f21 + f20 + f19 + f18 + f17 + f16 + f15 + f14 + f13 + f12 + f11 + f10 + f9 + f8 + f7 + f6 + f5 + f4 + f3 + f2 + f1 + f0)-1.0)/3.0;
}

CudaDeviceFunction vector_t getU(){
	real_t d = getRho();
	vector_t u;
u.x = f26 - f24 + f23 - f21 + f20 - f18 + f17 - f15 + f14 - f12 + f11 - f9 + f8 - f6 + f5 - f3 + f2 - f0;
u.y = f26 + f25 + f24 - f20 - f19 - f18 + f17 + f16 + f15 - f11 - f10 - f9 + f8 + f7 + f6 - f2 - f1 - f0;
u.z = f26 + f25 + f24 + f23 + f22 + f21 + f20 + f19 + f18 - f8 - f7 - f6 - f5 - f4 - f3 - f2 - f1 - f0;

	u.x /= d;
	u.y /= d;
	u.z /= d;
	return u;
}

CudaDeviceFunction float2 Color() {
        float2 ret;
        vector_t u = getU();
        ret.x = sqrt(u.x*u.x + u.y*u.y + u.z*u.z);
        if (NodeType == NODE_Solid){
                ret.y = 0;
        } else {
                ret.y = 1;
        }
        return ret;
}

CudaDeviceFunction void BounceBack()
{
real_t tmp;
tmp = f22;
f22 = f4;
f4 = tmp;
tmp = f10;
f10 = f16;
f16 = tmp;
tmp = f19;
f19 = f7;
f7 = tmp;
tmp = f1;
f1 = f25;
f25 = tmp;
tmp = f12;
f12 = f14;
f14 = tmp;
tmp = f21;
f21 = f5;
f5 = tmp;
tmp = f23;
f23 = f3;
f3 = tmp;
tmp = f11;
f11 = f15;
f15 = tmp;
tmp = f17;
f17 = f9;
f9 = tmp;
tmp = f18;
f18 = f8;
f8 = tmp;
tmp = f0;
f0 = f26;
f26 = tmp;
tmp = f20;
f20 = f6;
f6 = tmp;
tmp = f2;
f2 = f24;
f24 = tmp;

}

CudaDeviceFunction void SymmetryY()
{
real_t tmp;
tmp = f10;
f10 = f16;
f16 = tmp;
tmp = f19;
f19 = f25;
f25 = tmp;
tmp = f1;
f1 = f7;
f7 = tmp;
tmp = f15;
f15 = f9;
f9 = tmp;
tmp = f11;
f11 = f17;
f17 = tmp;
tmp = f18;
f18 = f24;
f24 = tmp;
tmp = f0;
f0 = f6;
f6 = tmp;
tmp = f20;
f20 = f26;
f26 = tmp;
tmp = f2;
f2 = f8;
f8 = tmp;

}

CudaDeviceFunction void SymmetryZ()
{
real_t tmp;
tmp = f22;
f22 = f4;
f4 = tmp;
tmp = f1;
f1 = f19;
f19 = tmp;
tmp = f25;
f25 = f7;
f7 = tmp;
tmp = f21;
f21 = f3;
f3 = tmp;
tmp = f23;
f23 = f5;
f5 = tmp;
tmp = f0;
f0 = f18;
f18 = tmp;
tmp = f24;
f24 = f6;
f6 = tmp;
tmp = f2;
f2 = f20;
f20 = tmp;
tmp = f26;
f26 = f8;
f8 = tmp;

}

CudaDeviceFunction void EVelocity()
{
real_t Jx, Jy, Jz, rho;
rho  = ( f25 + f22 + f19 + f16 + f13 + f10 + f7 + f4 + f1 + ( f14 + f11 + f17 + f8 + f20 + f5 + f23 + f2 + f26 )*2. ) / ( 1 + Velocity );
Jx = Velocity*rho;
Jy  = ( f25 - f19 + f16 - f10 + f7 - f1 ) / ( -1/3. );
Jz  = ( f25 + f22 + f19 - f7 - f4 - f1 ) / ( -1/3. );
f0 = f26 + ( -Jz - Jy - Jx )/36.;
f3 = f23 + ( -Jz - Jx )/9.;
f6 = f20 + ( -Jz + Jy - Jx )/36.;
f9 = f17 + ( -Jy - Jx )/9.;
f12 = f14 - Jx*4./9.;
f15 = f11 + ( Jy - Jx )/9.;
f18 = f8 + ( Jz - Jy - Jx )/36.;
f21 = f5 + ( Jz - Jx )/9.;
f24 = f2 + ( Jz + Jy - Jx )/36.;

}

CudaDeviceFunction void WPressure()
{
real_t Jx, Jy, Jz, rho;
rho = 1 + Pressure*3.;
Jx  = ( f25 + f22 + f19 + f16 + f13 + f10 + f7 + f4 - rho + f1 + ( f12 + f15 + f9 + f18 + f6 + f21 + f3 + f24 + f0 )*2. ) / ( -1 );
Jy  = ( f25 - f19 + f16 - f10 + f7 - f1 ) / ( -1/3. );
Jz  = ( f25 + f22 + f19 - f7 - f4 - f1 ) / ( -1/3. );
f2 = f24 + ( -Jz - Jy + Jx )/36.;
f5 = f21 + ( -Jz + Jx )/9.;
f8 = f18 + ( -Jz + Jy + Jx )/36.;
f11 = f15 + ( -Jy + Jx )/9.;
f14 = f12 + Jx*4./9.;
f17 = f9 + ( Jy + Jx )/9.;
f20 = f6 + ( Jz - Jy + Jx )/36.;
f23 = f3 + ( Jz + Jx )/9.;
f26 = f0 + ( Jz + Jy + Jx )/36.;

}

CudaDeviceFunction void WVelocity()
{
real_t Jx, Jy, Jz, rho;
rho  = ( f25 + f22 + f19 + f16 + f13 + f10 + f7 + f4 + f1 + ( f12 + f15 + f9 + f18 + f6 + f21 + f3 + f24 + f0 )*2. ) / ( 1 - Velocity );
Jx = Velocity*rho;
Jy  = ( f25 - f19 + f16 - f10 + f7 - f1 ) / ( -1/3. );
Jz  = ( f25 + f22 + f19 - f7 - f4 - f1 ) / ( -1/3. );
f2 = f24 + ( -Jz - Jy + Jx )/36.;
f5 = f21 + ( -Jz + Jx )/9.;
f8 = f18 + ( -Jz + Jy + Jx )/36.;
f11 = f15 + ( -Jy + Jx )/9.;
f14 = f12 + Jx*4./9.;
f17 = f9 + ( Jy + Jx )/9.;
f20 = f6 + ( Jz - Jy + Jx )/36.;
f23 = f3 + ( Jz + Jx )/9.;
f26 = f0 + ( Jz + Jy + Jx )/36.;

}

CudaDeviceFunction void EPressure()
{
real_t Jx, Jy, Jz, rho;
rho = 1 + Pressure*3.;
Jx  = ( f25 + f22 + f19 + f16 + f13 + f10 + f7 + f4 + f1 - rho + ( f14 + f11 + f17 + f8 + f20 + f5 + f23 + f2 + f26 )*2. ) / ( 1 );
Jy  = ( f25 - f19 + f16 - f10 + f7 - f1 ) / ( -1/3. );
Jz  = ( f25 + f22 + f19 - f7 - f4 - f1 ) / ( -1/3. );
f0 = f26 + ( -Jz - Jy - Jx )/36.;
f3 = f23 + ( -Jz - Jx )/9.;
f6 = f20 + ( -Jz + Jy - Jx )/36.;
f9 = f17 + ( -Jy - Jx )/9.;
f12 = f14 - Jx*4./9.;
f15 = f11 + ( Jy - Jx )/9.;
f18 = f8 + ( Jz - Jy - Jx )/36.;
f21 = f5 + ( Jz - Jx )/9.;
f24 = f2 + ( Jz + Jy - Jx )/36.;

}

CudaDeviceFunction void Run() {
    switch (NodeType & NODE_BOUNDARY) {
	case NODE_WPressure:
		WPressure();
		break;
	case NODE_WVelocity:
		WVelocity();
		break;
	case NODE_EPressure:
		EPressure();
		break;
	case NODE_SymmetryY:
		SymmetryY();
		break;
	case NODE_SymmetryZ:
		SymmetryZ();
		break;
	case NODE_Wall:
		BounceBack();
                break;
    }
    switch (NodeType & NODE_COLLISION) {
	case NODE_MRT:
		CollisionMRT();
		break;
    }
}

CudaDeviceFunction void SetEquilibrum(real_t rho, real_t Jx, real_t Jy, real_t Jz)
{
	f0 = ( rho + ( -Jz - Jy - Jx + ( Jz*Jz + Jy*Jy + Jx*Jx + ( Jz*Jy + ( Jz + Jy )*Jx )*3. )/rho )*3. )*0.00462962962962963;
f1 = rho*0.0185185185185185 + ( -Jx*Jx/rho + ( -Jz - Jy + ( Jz*Jz + ( Jy + Jz*3. )*Jy )/rho )*2. )/36.;
f2 = ( rho + ( -Jz - Jy + Jx + ( Jz*Jz + Jy*Jy + Jx*Jx + ( Jz*Jy + ( -Jz - Jy )*Jx )*3. )/rho )*3. )*0.00462962962962963;
f3 = rho*0.0185185185185185 + ( -Jy*Jy/rho + ( -Jz - Jx + ( Jz*Jz + ( Jx + Jz*3. )*Jx )/rho )*2. )/36.;
f4 = ( rho*2. + ( -Jz*2. + ( -Jx*Jx - Jy*Jy + Jz*Jz*2. )/rho )*3. )/27.;
f5 = rho*0.0185185185185185 + ( -Jy*Jy/rho + ( -Jz + Jx + ( Jz*Jz + ( Jx - Jz*3. )*Jx )/rho )*2. )/36.;
f6 = ( rho + ( -Jz + Jy - Jx + ( Jz*Jz + Jy*Jy + Jx*Jx + ( -Jz*Jy + ( Jz - Jy )*Jx )*3. )/rho )*3. )*0.00462962962962963;
f7 = rho*0.0185185185185185 + ( -Jx*Jx/rho + ( -Jz + Jy + ( Jz*Jz + ( Jy - Jz*3. )*Jy )/rho )*2. )/36.;
f8 = ( rho + ( -Jz + Jy + Jx + ( Jz*Jz + Jy*Jy + Jx*Jx + ( -Jz*Jy + ( -Jz + Jy )*Jx )*3. )/rho )*3. )*0.00462962962962963;
f9 = rho*0.0185185185185185 + ( -Jz*Jz/rho + ( -Jy - Jx + ( Jy*Jy + ( Jx + Jy*3. )*Jx )/rho )*2. )/36.;
f10 = ( rho*2. + ( -Jy*2. + ( -Jx*Jx - Jz*Jz + Jy*Jy*2. )/rho )*3. )/27.;
f11 = rho*0.0185185185185185 + ( -Jz*Jz/rho + ( -Jy + Jx + ( Jy*Jy + ( Jx - Jy*3. )*Jx )/rho )*2. )/36.;
f12 = ( rho*2. + ( -Jx*2. + ( -Jy*Jy - Jz*Jz + Jx*Jx*2. )/rho )*3. )/27.;
f13 = ( rho*2. + ( -Jx*Jx - Jy*Jy - Jz*Jz )/rho*3. )*4./27.;
f14 = ( rho*2. + ( Jx*2. + ( -Jy*Jy - Jz*Jz + Jx*Jx*2. )/rho )*3. )/27.;
f15 = rho*0.0185185185185185 + ( -Jz*Jz/rho + ( Jy - Jx + ( Jy*Jy + ( Jx - Jy*3. )*Jx )/rho )*2. )/36.;
f16 = ( rho*2. + ( Jy*2. + ( -Jx*Jx - Jz*Jz + Jy*Jy*2. )/rho )*3. )/27.;
f17 = rho*0.0185185185185185 + ( -Jz*Jz/rho + ( Jy + Jx + ( Jy*Jy + ( Jx + Jy*3. )*Jx )/rho )*2. )/36.;
f18 = ( rho + ( Jz - Jy - Jx + ( Jz*Jz + Jy*Jy + Jx*Jx + ( -Jz*Jy + ( -Jz + Jy )*Jx )*3. )/rho )*3. )*0.00462962962962963;
f19 = rho*0.0185185185185185 + ( -Jx*Jx/rho + ( Jz - Jy + ( Jz*Jz + ( Jy - Jz*3. )*Jy )/rho )*2. )/36.;
f20 = ( rho + ( Jz - Jy + Jx + ( Jz*Jz + Jy*Jy + Jx*Jx + ( -Jz*Jy + ( Jz - Jy )*Jx )*3. )/rho )*3. )*0.00462962962962963;
f21 = rho*0.0185185185185185 + ( -Jy*Jy/rho + ( Jz - Jx + ( Jz*Jz + ( Jx - Jz*3. )*Jx )/rho )*2. )/36.;
f22 = ( rho*2. + ( Jz*2. + ( -Jx*Jx - Jy*Jy + Jz*Jz*2. )/rho )*3. )/27.;
f23 = rho*0.0185185185185185 + ( -Jy*Jy/rho + ( Jz + Jx + ( Jz*Jz + ( Jx + Jz*3. )*Jx )/rho )*2. )/36.;
f24 = ( rho + ( Jz + Jy - Jx + ( Jz*Jz + Jy*Jy + Jx*Jx + ( Jz*Jy + ( -Jz - Jy )*Jx )*3. )/rho )*3. )*0.00462962962962963;
f25 = rho*0.0185185185185185 + ( -Jx*Jx/rho + ( Jz + Jy + ( Jz*Jz + ( Jy + Jz*3. )*Jy )/rho )*2. )/36.;
f26 = ( rho + ( Jz + Jy + Jx + ( Jz*Jz + Jy*Jy + Jx*Jx + ( Jz*Jy + ( Jz + Jy )*Jx )*3. )/rho )*3. )*0.00462962962962963;

}

CudaDeviceFunction void Init() {
	vector_t ST = SyntheticTurbulence(0);
	ST.x = Velocity + Turbulence*ST.x;
	ST.y = Turbulence*ST.y;
	ST.z = Turbulence*ST.z;
	SetEquilibrum(1.0 + Pressure * 3.0, ST.x, ST.y, ST.z);
}

CudaDeviceFunction void CollisionMRT()
{

	real_t rho,Jx,Jy,Jz,R4,R5,R6,R7,R8,R9,R10,R11,R12,R13,R14,R15,R16,R17,R18,R19,R20,R21,R22,R23,R24,R25,R26;
	real_t gamma  = 1-omega;
rho = f26 + f25 + f24 + f23 + f22 + f21 + f20 + f19 + f18 + f17 + f16 + f15 + f14 + f13 + f12 + f11 + f10 + f9 + f8 + f7 + f6 + f5 + f4 + f3 + f2 + f1 + f0;
Jx = f26 - f24 + f23 - f21 + f20 - f18 + f17 - f15 + f14 - f12 + f11 - f9 + f8 - f6 + f5 - f3 + f2 - f0;
Jy = f26 + f25 + f24 - f20 - f19 - f18 + f17 + f16 + f15 - f11 - f10 - f9 + f8 + f7 + f6 - f2 - f1 - f0;
Jz = f26 + f25 + f24 + f23 + f22 + f21 + f20 + f19 + f18 - f8 - f7 - f6 - f5 - f4 - f3 - f2 - f1 - f0;
R4 = f26 + f25 + f24 + f23 + f22 + f21 + f20 + f19 + f18 + f8 + f7 + f6 + f5 + f4 + f3 + f2 + f1 + f0;
R5 = f26 + f25 + f24 + f20 + f19 + f18 + f17 + f16 + f15 + f11 + f10 + f9 + f8 + f7 + f6 + f2 + f1 + f0;
R6 = f26 + f24 + f23 + f21 + f20 + f18 + f17 + f15 + f14 + f12 + f11 + f9 + f8 + f6 + f5 + f3 + f2 + f0;
R7 = f26 - f24 - f20 + f18 + f17 - f15 - f11 + f9 + f8 - f6 - f2 + f0;
R8 = f26 - f24 + f23 - f21 + f20 - f18 - f8 + f6 - f5 + f3 - f2 + f0;
R9 = f26 + f25 + f24 - f20 - f19 - f18 - f8 - f7 - f6 + f2 + f1 + f0;
R10 = f26 - f24 + f23 - f21 + f20 - f18 + f8 - f6 + f5 - f3 + f2 - f0;
R11 = f26 + f25 + f24 - f20 - f19 - f18 + f8 + f7 + f6 - f2 - f1 - f0;
R12 = f26 - f24 + f20 - f18 + f17 - f15 + f11 - f9 + f8 - f6 + f2 - f0;
R13 = f26 + f24 - f20 - f18 + f17 + f15 - f11 - f9 + f8 + f6 - f2 - f0;
R14 = f26 + f25 + f24 + f20 + f19 + f18 - f8 - f7 - f6 - f2 - f1 - f0;
R15 = f26 + f24 + f23 + f21 + f20 + f18 - f8 - f6 - f5 - f3 - f2 - f0;
R16 = f26 - f24 - f20 + f18 - f8 + f6 + f2 - f0;
R17 = f26 + f25 + f24 + f20 + f19 + f18 + f8 + f7 + f6 + f2 + f1 + f0;
R18 = f26 + f24 + f23 + f21 + f20 + f18 + f8 + f6 + f5 + f3 + f2 + f0;
R19 = f26 - f24 - f20 + f18 + f8 - f6 - f2 + f0;
R20 = f26 + f24 + f20 + f18 + f17 + f15 + f11 + f9 + f8 + f6 + f2 + f0;
R21 = f26 - f24 + f20 - f18 - f8 + f6 - f2 + f0;
R22 = f26 + f24 - f20 - f18 - f8 - f6 + f2 + f0;
R23 = f26 - f24 + f20 - f18 + f8 - f6 + f2 - f0;
R24 = f26 + f24 - f20 - f18 + f8 + f6 - f2 - f0;
R25 = f26 + f24 + f20 + f18 - f8 - f6 - f2 - f0;
R26 = f26 + f24 + f20 + f18 + f8 + f6 + f2 + f0;
R4 = R4 - Jz*Jz/rho - rho/3.;
R5 = R5 - Jy*Jy/rho - rho/3.;
R6 = R6 - Jx*Jx/rho - rho/3.;
R7 = R7 - Jy*Jx/rho;
R8 = R8 - Jz*Jx/rho;
R9 = R9 - Jz*Jy/rho;
R10 = R10 - Jx/3.;
R11 = R11 - Jy/3.;
R12 = R12 - Jx/3.;
R13 = R13 - Jy/3.;
R14 = R14 - Jz/3.;
R15 = R15 - Jz/3.;
R16 = R16;
R17 = R17 + ( -rho + ( -Jz*Jz - Jy*Jy )/rho*3. )/9.;
R18 = R18 + ( -rho + ( -Jz*Jz - Jx*Jx )/rho*3. )/9.;
R19 = R19 - Jy*Jx/rho/3.;
R20 = R20 + ( -rho + ( -Jy*Jy - Jx*Jx )/rho*3. )/9.;
R21 = R21 - Jz*Jx/rho/3.;
R22 = R22 - Jz*Jy/rho/3.;
R23 = R23 - Jx/9.;
R24 = R24 - Jy/9.;
R25 = R25 - Jz/9.;
R26 = R26 + ( -rho + ( -Jz*Jz - Jy*Jy - Jx*Jx )/rho*3. )/27.;

	if (NodeType & NODE_LES) {
		real_t Q, tau, tau0;
Q = R4*R4 + R5*R5 + R6*R6 + ( R9*R9 + R7*R7 + R8*R8 )*2.;

		Q = 18.* sqrt(Q) * Smag;
		tau0 = 1/(1-gamma);
		tau = tau0*tau0 + Q;
		tau = sqrt(tau);
		tau = (tau + tau0)/2;
		gamma = 1. - 1./tau;
	}
	real_t gamma2 = gamma;
	if (NodeType & NODE_ENTROPIC) {
		real_t a,b;
a = ( -R17*R6 - R18*R5 - R20*R4 + ( -R22*R9 - R21*R8 - R19*R7 )*4. + ( ( R26 - R20 - R18 )*R6 + ( R26 - R20 - R17 )*R5 + ( R26 - R18 - R17 )*R4 )*3. )*27./8.;
b = R16*R16*27. + ( ( R15*R14 + R13*R11 + R12*R10 )*4. + ( ( R20*R20 + R18*R18 + R17*R17 + R26*R26*3. )*3. + ( R20*R18 + R15*R15 + R14*R14 + R13*R13 + R11*R11 + R12*R12 + R10*R10 + ( R20 + R18 )*R17 + ( R25*R25 + R24*R24 + R23*R23 + ( -R20 - R18 - R17 )*R26 )*3. + ( R22*R22 + R21*R21 + R19*R19 + ( -R15 - R14 )*R25 + ( -R13 - R11 )*R24 + ( -R12 - R10 )*R23 )*2. )*2. )*3. )*27./8.;

		gamma2 = - gamma2 * a/b;
	}
R4 = gamma*R4;
R5 = gamma*R5;
R6 = gamma*R6;
R7 = gamma*R7;
R8 = gamma*R8;
R9 = gamma*R9;
R10 = gamma2*R10;
R11 = gamma2*R11;
R12 = gamma2*R12;
R13 = gamma2*R13;
R14 = gamma2*R14;
R15 = gamma2*R15;
R16 = gamma2*R16;
R17 = gamma2*R17;
R18 = gamma2*R18;
R19 = gamma2*R19;
R20 = gamma2*R20;
R21 = gamma2*R21;
R22 = gamma2*R22;
R23 = gamma2*R23;
R24 = gamma2*R24;
R25 = gamma2*R25;
R26 = gamma2*R26;

	Jx += ForceX;
	Jy += ForceY;
	Jz += ForceZ;
	if ((NodeType & NODE_BOUNDARY) == NODE_Solid) {
		Jx = 0;
		Jy = 0;
		Jz = 0;
	}
R4 = R4 + Jz*Jz/rho + rho/3.;
R5 = R5 + Jy*Jy/rho + rho/3.;
R6 = R6 + Jx*Jx/rho + rho/3.;
R7 = R7 + Jy*Jx/rho;
R8 = R8 + Jz*Jx/rho;
R9 = R9 + Jz*Jy/rho;
R10 = R10 + Jx/3.;
R11 = R11 + Jy/3.;
R12 = R12 + Jx/3.;
R13 = R13 + Jy/3.;
R14 = R14 + Jz/3.;
R15 = R15 + Jz/3.;
R16 = R16;
R17 = R17 + ( rho + ( Jz*Jz + Jy*Jy )/rho*3. )/9.;
R18 = R18 + ( rho + ( Jz*Jz + Jx*Jx )/rho*3. )/9.;
R19 = R19 + Jy*Jx/rho/3.;
R20 = R20 + ( rho + ( Jy*Jy + Jx*Jx )/rho*3. )/9.;
R21 = R21 + Jz*Jx/rho/3.;
R22 = R22 + Jz*Jy/rho/3.;
R23 = R23 + Jx/9.;
R24 = R24 + Jy/9.;
R25 = R25 + Jz/9.;
R26 = R26 + ( rho + ( Jz*Jz + Jy*Jy + Jx*Jx )/rho*3. )/27.;
f0 = ( R26 - R25 - R24 - R23 + R22 + R21 + R19 - R16 )/8.;
f1 = ( -R26 + R25 + R24 - R22 + R17 - R14 - R11 + R9 )/4.;
f2 = ( R26 - R25 - R24 + R23 + R22 - R21 - R19 + R16 )/8.;
f3 = ( -R26 + R25 + R23 - R21 + R18 - R15 - R10 + R8 )/4.;
f4 = ( R26 - R25 - R18 - R17 + R15 + R14 + R4 - Jz )/2.;
f5 = ( -R26 + R25 - R23 + R21 + R18 - R15 + R10 - R8 )/4.;
f6 = ( R26 - R25 + R24 - R23 - R22 + R21 - R19 + R16 )/8.;
f7 = ( -R26 + R25 - R24 + R22 + R17 - R14 + R11 - R9 )/4.;
f8 = ( R26 - R25 + R24 + R23 - R22 - R21 + R19 - R16 )/8.;
f9 = ( -R26 + R24 + R23 + R20 - R19 - R13 - R12 + R7 )/4.;
f10 = ( R26 - R24 - R20 - R17 + R13 + R11 + R5 - Jy )/2.;
f11 = ( -R26 + R24 - R23 + R20 + R19 - R13 + R12 - R7 )/4.;
f12 = ( R26 - R23 - R20 - R18 + R12 + R10 + R6 - Jx )/2.;
f13 = -R26 + R20 + R18 + R17 - R6 - R5 - R4 + rho;
f14 = ( R26 + R23 - R20 - R18 - R12 - R10 + R6 + Jx )/2.;
f15 = ( -R26 - R24 + R23 + R20 + R19 + R13 - R12 - R7 )/4.;
f16 = ( R26 + R24 - R20 - R17 - R13 - R11 + R5 + Jy )/2.;
f17 = ( -R26 - R24 - R23 + R20 - R19 + R13 + R12 + R7 )/4.;
f18 = ( R26 + R25 - R24 - R23 - R22 - R21 + R19 + R16 )/8.;
f19 = ( -R26 - R25 + R24 + R22 + R17 + R14 - R11 - R9 )/4.;
f20 = ( R26 + R25 - R24 + R23 - R22 + R21 - R19 - R16 )/8.;
f21 = ( -R26 - R25 + R23 + R21 + R18 + R15 - R10 - R8 )/4.;
f22 = ( R26 + R25 - R18 - R17 - R15 - R14 + R4 + Jz )/2.;
f23 = ( -R26 - R25 - R23 - R21 + R18 + R15 + R10 + R8 )/4.;
f24 = ( R26 + R25 + R24 - R23 + R22 - R21 - R19 - R16 )/8.;
f25 = ( -R26 - R25 - R24 - R22 + R17 + R14 + R11 + R9 )/4.;
f26 = ( R26 + R25 + R24 + R23 + R22 + R21 + R19 + R16 )/8.;

}

