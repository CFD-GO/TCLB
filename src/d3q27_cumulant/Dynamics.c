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

         real_t m[27],c[27];
 	 real_t w[10] = {1.0/(3*nu+0.5),1.,1.,1.,1.,1.,1.,1.,1.,1.};
         for (int i = 0;; i < 27; i++) m[i] = f[i];         
//         Moment transform:
          
	m[0] = m[2] + m[1] + m[0];
	m[1] = -m[2] + m[1]; 
	m[2] = m[1] + m[2]*2.; 
	m[3] = m[8] + m[7] + m[3];	
	m[7] = -m[8] + m[7];
	m[8] = m[7] + m[8]*2.; 
	m[4] = m[10] + m[9] + m[4];
	m[9] = -m[10] + m[9]; 
	m[10] = m[9] + m[10]*2.; 
	m[5] = m[12] + m[11] + m[5];
	m[11] = -m[12] + m[11];
	m[12] = m[11] + m[12]*2.; 
	m[15] = m[20] + m[19] + m[15];
	m[19] = -m[20] + m[19]; 
	m[20] = m[19] + m[20]*2.; 
	m[16] = m[22] + m[21] + m[16];
	m[21] = -m[22] + m[21];
	m[22] = m[21] + m[22]*2.; 
	m[6] = m[14] + m[13] + m[6]; 
	m[13] = -m[14] + m[13]; 
	m[14] = m[13] + m[14]*2.; 
	m[17] = m[24] + m[23] + m[17];
	m[23] = -m[24] + m[23];
	m[24] = m[23] + m[24]*2.; 
	m[18] = m[26] + m[25] + m[18]; 
	m[25] = -m[26] + m[25];
	m[26] = m[25] + m[26]*2.; 
	m[0] = m[4] + m[3] + m[0];
	m[3] = -m[4] + m[3];
	m[4] = m[3] + m[4]*2.; 
	m[1] = m[9] + m[7] + m[1]; 
	m[7] = -m[9] + m[7]; 
	m[9] = m[7] + m[9]*2.; 
	m[2] = m[10] + m[8] + m[2];
	m[8] = -m[10] + m[8]; 
	m[10] = m[8] + m[10]*2.; 
	m[5] = m[16] + m[15] + m[5];
	m[15] = -m[16] + m[15];
	m[16] = m[15] + m[16]*2.; 
	m[11] = m[21] + m[19] + m[11];
	m[19] = -m[21] + m[19]; 
	m[21] = m[19] + m[21]*2.; 
	m[12] = m[22] + m[20] + m[12];
	m[20] = -m[22] + m[20]; 
	m[22] = m[20] + m[22]*2.; 
	m[6] = m[18] + m[17] + m[6];
	m[17] = -m[18] + m[17];
	m[18] = m[17] + m[18]*2.; 
	m[13] = m[25] + m[23] + m[13];
	m[23] = -m[25] + m[23]; 
	m[25] = m[23] + m[25]*2.; 
	m[14] = m[26] + m[24] + m[14]; 
	m[24] = -m[26] + m[24];
	m[26] = m[24] + m[26]*2.; 
	m[0] = m[6] + m[5] + m[0]; 
	m[5] = -m[6] + m[5]; 
	m[6] = m[5] + m[6]*2.; 
	m[1] = m[13] + m[11] + m[1]; 
	m[11] = -m[13] + m[11];
	m[13] = m[11] + m[13]*2.; 
	m[2] = m[14] + m[12] + m[2];
	m[12] = -m[14] + m[12]; 
	m[14] = m[12] + m[14]*2.; 
	m[3] = m[17] + m[15] + m[3];
	m[15] = -m[17] + m[15]; 
	m[17] = m[15] + m[17]*2.; 
	m[7] = m[23] + m[19] + m[7];
	m[19] = -m[23] + m[19]; 
	m[23] = m[19] + m[23]*2.; 
	m[8] = m[24] + m[20] + m[8];
	m[20] = -m[24] + m[20]; 
	m[24] = m[20] + m[24]*2.; 
	m[4] = m[18] + m[16] + m[4]; 
	m[16] = -m[18] + m[16];
	m[18] = m[16] + m[18]*2.; 
	m[9] = m[25] + m[21] + m[9]; 
	m[21] = -m[25] + m[21];
	m[25] = m[21] + m[25]*2.; 
	m[10] = m[26] + m[22] + m[10];
	m[22] = -m[26] + m[22]; 
	m[26] = m[22] + m[26]*2.; 




//Moment to Cummulant-Transform:

	c[0] = m[0]
	c[1] = m[1]/m[0];
	c[2] = ( -c[1]*m[1] + m[2] )/m[0];
	c[3] = m[3]/m[0];
	c[7] = ( -c[1]*m[3] + m[7] )/m[0];
	c[8] = ( -c[7]*m[1] - c[2]*m[3] - c[1]*m[7] + m[8] )/m[0];
	c[4] = ( -c[3]*m[3] + m[4] )/m[0];
	c[9] = ( -c[1]*m[4] + m[9] - c[7]*m[3]*2. )/m[0];
	c[10] = ( -c[9]*m[1] - c[2]*m[4] - c[1]*m[9] + m[10] + ( -c[8]*m[3] - c[7]*m[7] )*2. )/m[0];
	c[5] = m[5]/m[0];
	c[11] = ( -c[1]*m[5] + m[11] )/m[0];
	c[12] = ( -c[11]*m[1] - c[2]*m[5] - c[1]*m[11] + m[12] )/m[0];
	c[15] = ( -c[3]*m[5] + m[15] )/m[0];
	c[19] = ( -c[11]*m[3] - c[7]*m[5] - c[1]*m[15] + m[19] )/m[0];
	c[20] = ( -c[15]*m[2] - c[8]*m[5] - c[3]*m[12] + m[20] + ( -c[19]*m[1] - c[7]*m[11] )*2. )/m[0];
	c[16] = ( -c[15]*m[3] - c[4]*m[5] - c[3]*m[15] + m[16] )/m[0];
	c[21] = ( -c[11]*m[4] - c[9]*m[5] - c[1]*m[16] + m[21] + ( -c[19]*m[3] - c[7]*m[15] )*2. )/m[0];
	c[22] = ( -c[16]*m[2] - c[12]*m[4] - c[5]*m[10] + m[22] + ( -c[21]*m[1] - c[20]*m[3] - c[15]*m[8] - c[11]*m[9] - c[19]*m[7]*2. )*2. )/m[0];
	c[6] = ( -c[5]*m[5] + m[6] )/m[0];
	c[13] = ( -c[1]*m[6] + m[13] - c[11]*m[5]*2. )/m[0];
	c[14] = ( -c[13]*m[1] - c[2]*m[6] - c[1]*m[13] + m[14] + ( -c[12]*m[5] - c[11]*m[11] )*2. )/m[0];
	c[17] = ( -c[3]*m[6] + m[17] - c[15]*m[5]*2. )/m[0];
	c[23] = ( -c[13]*m[3] - c[7]*m[6] - c[1]*m[17] + m[23] + ( -c[19]*m[5] - c[11]*m[15] )*2. )/m[0];
	c[24] = ( -c[17]*m[2] - c[8]*m[6] - c[3]*m[14] + m[24] + ( -c[23]*m[1] - c[20]*m[5] - c[15]*m[12] - c[7]*m[13] - c[19]*m[11]*2. )*2. )/m[0];
	c[18] = ( -c[17]*m[3] - c[4]*m[6] - c[3]*m[17] + m[18] + ( -c[16]*m[5] - c[15]*m[15] )*2. )/m[0];
	c[25] = ( -c[13]*m[4] - c[9]*m[6] - c[1]*m[18] + m[25] + ( -c[23]*m[3] - c[21]*m[5] - c[11]*m[16] - c[7]*m[17] - c[19]*m[15]*2. )*2. )/m[0];
	c[26] = ( -c[25]*m[1] - c[14]*m[4] - c[13]*m[9] - c[10]*m[6] - c[9]*m[13] - c[2]*m[18] - c[1]*m[25] + m[26] + ( -c[24]*m[3] - c[23]*m[7] - c[22]*m[5] - c[21]*m[11] - c[12]*m[16]
 - c[11]*m[21] - c[8]*m[17] - c[7]*m[23] + ( -c[20]*m[15] - c[19]*m[19] )*2. )*2. )/m[0];
//Galilean correction terms definition: 
       real_t dxu,dyv,dzw;
       dxu = - w[0]/(2.)*(2*c[2] - c[4] - c[6]) - w[1]/(2.)*(c[2] + c[4] + c[6] - d);
       dyv = dxu + 3*w[0]/2.*(c[2] - c[4]);
       dzw = dxu + 3*w[0]/2.*(c[2] - c[6]);
       real_t a,b,c;
       a = (1 - w[0])*(c[2] - c[4]);
       b = (1 - w[0])*(c[2] - c[6]);
       c = w[1] + (1 - w[1])*(c[2] + c[4] + c[6]);
//Cumulants relation 

	c[0] = c[0];//000  no change
 	c[1] = c[1];//100 no change
        c[2] = (a + b + c)/3.; // 200
        c[4] =(c - 2*a + b)/3.;//020
        c[6] = (c - 2*b + a)/3.; 
        c[3] = c[3]; //010 no change
        c[5] = c[5]; //001 no change
        c[7] = (1 - w[0])*c[7];//110
        c[8] =((1 - w[2])*(c[8] + c[17]) + (1 - w[3])*(c[8] - c[17]))/2.;//210
        c[17] = ((1 - w[2])*(c[8] + c[17]) - (1 - w[3])*(c[8] - c[17]))/2.;//012
        c[9] = ((1 - w[2])*(c[9] + c[13]) + (1 - w[3])*(c[9] - c[13]))/2.;//120
        c[13] = ((1 - w[2])*(c[9] + c[13]) - (1 - w[3])*(c[9] - c[13]))/2.;//102
        c[11] = (1 - w[0])*c[11];//101
        c[12] = ((1 - w[2])*(c[12] + c[16]) + (1 - w[3])*(c[12] - c[16]))/2.; //201
        c[16] = ((1 - w[2])*(c[12] + c[16]) - (1 - w[3])*(c[12] - c[16]))/2.; //021
        c[15] = (1 - w[0])*c[15];//011
        a = (1 - w[5])*(c[10] - 2*c[14] + c[18]);
        b = (1 - w[5])*(c[10] - 2*c[18] + c[14]);
        c = (1 - w[6])*(c[10] + c[14] + c[18]);
        c[10] = (a + b + c)/3.; //220
        c[14] =  (c - a)/3.;//202
        c[18] = (c - b)/3.; //022
        c[19] = (1 - w[4])*c[19]//111
        c[20] = (1 - w[7])*c[20];
        c[21] = (1 - w[7])*c[21];
        c[22] = (1 - w[8])*c[22];
        c[23] = (1 - w[7])*c[23];
        c[24] = (1 - w[8])*c[24];
        c[25] = (1 - w[8])*c[25];
        c[26] = (1 - w[9])*c[26];


//Cummulant -moment transform

	m[0] = m[0];
	m[1] = c[1]*m[0];
	m[2] = c[2]*m[0] + c[1]*m[1];
	m[3] = c[3]*m[0];
	m[7] = c[7]*m[0] + c[1]*m[3];
	m[8] = c[8]*m[0] + c[7]*m[1] + c[2]*m[3] + c[1]*m[7];
	m[4] = c[4]*m[0] + c[3]*m[3];
	m[9] = c[9]*m[0] + c[1]*m[4] + c[7]*m[3]*2.;
	m[10] = c[10]*m[0] + c[9]*m[1] + c[2]*m[4] + c[1]*m[9] + ( c[8]*m[3] + c[7]*m[7] )*2.;
	m[5] = c[5]*m[0];
	m[11] = c[11]*m[0] + c[1]*m[5];
	m[12] = c[12]*m[0] + c[11]*m[1] + c[2]*m[5] + c[1]*m[11];
	m[15] = c[15]*m[0] + c[3]*m[5];
	m[19] = c[19]*m[0] + c[11]*m[3] + c[7]*m[5] + c[1]*m[15];
	m[20] = c[20]*m[0] + c[15]*m[2] + c[8]*m[5] + c[3]*m[12] + ( c[19]*m[1] + c[7]*m[11] )*2.;
	m[16] = c[16]*m[0] + c[15]*m[3] + c[4]*m[5] + c[3]*m[15];
	m[21] = c[21]*m[0] + c[11]*m[4] + c[9]*m[5] + c[1]*m[16] + ( c[19]*m[3] + c[7]*m[15] )*2.;
	m[22] = c[22]*m[0] + c[16]*m[2] + c[12]*m[4] + c[5]*m[10] + ( c[21]*m[1] + c[20]*m[3] + c[15]*m[8] + c[11]*m[9] + c[19]*m[7]*2. )*2.;
	m[6] = c[6]*m[0] + c[5]*m[5];
	m[13] = c[13]*m[0] + c[1]*m[6] + c[11]*m[5]*2.;
	m[14] = c[14]*m[0] + c[13]*m[1] + c[2]*m[6] + c[1]*m[13] + ( c[12]*m[5] + c[11]*m[11] )*2.;
	m[17] = c[17]*m[0] + c[3]*m[6] + c[15]*m[5]*2.;
	m[23] = c[23]*m[0] + c[13]*m[3] + c[7]*m[6] + c[1]*m[17] + ( c[19]*m[5] + c[11]*m[15] )*2.;
	m[24] = c[24]*m[0] + c[17]*m[2] + c[8]*m[6] + c[3]*m[14] + ( c[23]*m[1] + c[20]*m[5] + c[15]*m[12] + c[7]*m[13] + c[19]*m[11]*2. )*2.;
	m[18] = c[18]*m[0] + c[17]*m[3] + c[4]*m[6] + c[3]*m[17] + ( c[16]*m[5] + c[15]*m[15] )*2.;
	m[25] = c[25]*m[0] + c[13]*m[4] + c[9]*m[6] + c[1]*m[18] + ( c[23]*m[3] + c[21]*m[5] + c[11]*m[16] + c[7]*m[17] + c[19]*m[15]*2. )*2.;
	m[26] = c[26]*m[0] + c[25]*m[1] + c[14]*m[4] + c[13]*m[9] + c[10]*m[6] + c[9]*m[13] + c[2]*m[18] + c[1]*m[25] + ( c[24]*m[3] + c[23]*m[7] + c[22]*m[5] + c[21]*m[11] + c[12]*m[16]
 + c[11]*m[21] + c[8]*m[17] + c[7]*m[23] + ( c[20]*m[15] + c[19]*m[19] )*2. )*2.


//Reverse moment transform:


	m[0] = -m[2] + m[0]; 
	m[1] = ( m[2] + m[1] )/2.;
	m[2] = m[2] - m[1]; 
	m[3] = -m[8] + m[3]; 
	m[7] = ( m[8] + m[7] )/2.;
	m[8] = m[8] - m[7]; 
	m[4] = -m[10] + m[4]; 
	m[9] = ( m[10] + m[9] )/2.;
	m[10] = m[10] - m[9]; 
	m[5] = -m[12] + m[5]; 
	m[11] = ( m[12] + m[11] )/2.;
	m[12] = m[12] - m[11]; 
	m[15] = -m[20] + m[15]; 
	m[19] = ( m[20] + m[19] )/2.;
	m[20] = m[20] - m[19]; 
	m[16] = -m[22] + m[16];
	m[21] = ( m[22] + m[21] )/2.;
	m[22] = m[22] - m[21]; 
	m[6] = -m[14] + m[6];
	m[13] = ( m[14] + m[13] )/2.;
	m[14] = m[14] - m[13]; 
	m[17] = -m[24] + m[17]; 
	m[23] = ( m[24] + m[23] )/2.;
	m[24] = m[24] - m[23]; 
	m[18] = -m[26] + m[18];
	m[25] = ( m[26] + m[25] )/2.;
	m[26] = m[26] - m[25]; 
	m[0] = -m[4] + m[0]; 
	m[3] = ( m[4] + m[3] )/2.;
	m[4] = m[4] - m[3]; 
	m[1] = -m[9] + m[1]; 
	m[7] = ( m[9] + m[7] )/2.;
	m[9] = m[9] - m[7]; 
	m[2] = -m[10] + m[2]; 
	m[8] = ( m[10] + m[8] )/2.;
	m[10] = m[10] - m[8]; 
	m[5] = -m[16] + m[5]; 
	m[15] = ( m[16] + m[15] )/2.;
	m[16] = m[16] - m[15]; 
	m[11] = -m[21] + m[11]; 
	m[19] = ( m[21] + m[19] )/2.;
	m[21] = m[21] - m[19]; 
	m[12] = -m[22] + m[12]; 
	m[20] = ( m[22] + m[20] )/2.;
	m[22] = m[22] - m[20]; 
	m[6] = -m[18] + m[6];
	m[17] = ( m[18] + m[17] )/2.;
	m[18] = m[18] - m[17]; 
	m[13] = -m[25] + m[13];
	m[23] = ( m[25] + m[23] )/2.;
	m[25] = m[25] - m[23]; 
	m[14] = -m[26] + m[14]; 
	m[24] = ( m[26] + m[24] )/2.; 
	m[26] = m[26] - m[24]; 
	m[0] = -m[6] + m[0];
	m[5] = ( m[6] + m[5] )/2.;
	m[6] = m[6] - m[5]; 
	m[1] = -m[13] + m[1];
	m[11] = ( m[13] + m[11] )/2.;
	m[13] = m[13] - m[11]; 
	m[2] = -m[14] + m[2];
	m[12] = ( m[14] + m[12] )/2.;
	m[14] = m[14] - m[12]; 
	m[3] = -m[17] + m[3]; 
	m[15] = ( m[17] + m[15] )/2.;
	m[17] = m[17] - m[15]; 
	m[7] = -m[23] + m[7]; 
	m[19] = ( m[23] + m[19] )/2.;
	m[23] = m[23] - m[19]; 
	m[8] = -m[24] + m[8];
	m[20] = ( m[24] + m[20] )/2.; 
	m[24] = m[24] - m[20]; 
	m[4] = -m[18] + m[4];
	m[16] = ( m[18] + m[16] )/2.;
	m[18] = m[18] - m[16]; 
	m[9] = -m[25] + m[9];
	m[21] = ( m[25] + m[21] )/2.; 
	m[25] = m[25] - m[21]; 
	m[10] = -m[26] + m[10]; 
	m[22] = ( m[26] + m[22] )/2.;
	m[26] = m[26] - m[22]; 
           

        for (int i = 0;; i < 27; i++) m[i] = f[i];         
	
 
}

