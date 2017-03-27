/*-------------------------------------------------------------*/
/*  	CLB - Cudne LB - Stencil Version                           */
/*     CUDA based Adjoint Lattice Boltzmann Solver             */
/*     Author: Wojciech Regulski                          */
/*     Author e-mail wojtek.regulski@gmail.com
/*     Developed at: Warsaw University of Technology - 2012    */
/*
	Description:
	This is the standard implementation of the BGK collision model [1] 
	- a very basic model used in the kinetic theory and th LBM framework 
 	The force term is added in a way propsed by Kuperstokh [2]:

	delta f_force = f_eq(u+du) - f_eq(u) where du = u+force/rho

	Mind that we add load to the momentum, J, thus no division by rho takes place in our implementation.

	References:
	[1]    Bhatnagar, P.L., Gross, E.P., Krook, M., 1954. A model for collision processes in gases. I. small amplitude processes in charged and neutral one-component systems.
	Phys. Rev. 94, 511-525. URL http://	link.aps.org/doi/10.1103/PhysRev.94.511.

	[2]	Kupershtokh, A., Medvedev, D., Karpov, D., 2009. "On equations of state in a lattice Boltzmann method". Comput. Math. Appl. 58 (5), 965-974


/*-------------------------------------------------------------*/


CudaDeviceFunction real_t getRho(){
	return f222 + f122 + f022 + f212 + f112 + f012 + f202 + f102 + f002 + f221 + f121 + f021 + f211 + f111 + f011 + f201 + f101 + f001 + f220 + f120 + f020 + f210 + f110 + f010 + f200 + f100 + f000;
}

CudaDeviceFunction real_t getP(){
	return ((f222 + f122 + f022 + f212 + f112 + f012 + f202 + f102 + f002 + f221 + f121 + f021 + f211 + f111 + f011 + f201 + f101 + f001 + f220 + f120 + f020 + f210 + f110 + f010 + f200 + f100 + f000)-1.0)/3.0;
}

CudaDeviceFunction vector_t getU(){
	real_t d = getRho();
	vector_t u;
u.x = -f222 + f122 - f212 + f112 - f202 + f102 - f221 + f121 - f211 + f111 - f201 + f101 - f220 + f120 - f210 + f110 - f200 + f100;
u.y = -f222 - f122 - f022 + f212 + f112 + f012 - f221 - f121 - f021 + f211 + f111 + f011 - f220 - f120 - f020 + f210 + f110 + f010;
u.z = -f222 - f122 - f022 - f212 - f112 - f012 - f202 - f102 - f002 + f221 + f121 + f021 + f211 + f111 + f011 + f201 + f101 + f001;

	u.x = (u.x + ForceX/2.)/d;
	u.y = (u.y + ForceY/2.)/d;
	u.z = (u.z + ForceZ/2.)/d;
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
tmp = f001;
f001 = f002;
f002 = tmp;
tmp = f010;
f010 = f020;
f020 = tmp;
tmp = f011;
f011 = f022;
f022 = tmp;
tmp = f012;
f012 = f021;
f021 = tmp;
tmp = f100;
f100 = f200;
f200 = tmp;
tmp = f101;
f101 = f202;
f202 = tmp;
tmp = f102;
f102 = f201;
f201 = tmp;
tmp = f120;
f120 = f210;
f210 = tmp;
tmp = f110;
f110 = f220;
f220 = tmp;
tmp = f121;
f121 = f212;
f212 = tmp;
tmp = f122;
f122 = f211;
f211 = tmp;
tmp = f111;
f111 = f222;
f222 = tmp;
tmp = f112;
f112 = f221;
f221 = tmp;

}

CudaDeviceFunction void SymmetryY()
{
real_t tmp;
tmp = f010;
f010 = f020;
f020 = tmp;
tmp = f011;
f011 = f021;
f021 = tmp;
tmp = f012;
f012 = f022;
f022 = tmp;
tmp = f210;
f210 = f220;
f220 = tmp;
tmp = f110;
f110 = f120;
f120 = tmp;vector_t u;
u.x = -f222 + f122 - f212 + f112 - f202 + f102 - f221 + f121 - f211 + f111 - f201 + f101 - f220 + f120 - f210 + f110 - f200 + f100;
u.y = -f222 - f122 - f022 + f212 + f112 + f012 - f221 - f121 - f021 + f211 + f111 + f011 - f220 - f120 - f020 + f210 + f110 + f010;
u.z = -f222 - f122 - f022 - f212 - f112 - f012 - f202 - f102 - f002 + f221 + f121 + f021 + f211 + f111 + f011 + f201 + f101 + f001;
tmp = f211;
f211 = f221;
f221 = tmp;
tmp = f212;
f212 = f222;
f222 = tmp;
tmp = f111;
f111 = f121;
f121 = tmp;
tmp = f112;
f112 = f122;
f122 = tmp;

}

CudaDeviceFunction void SymmetryZ()
{
real_t tmp;
tmp = f001;
f001 = f002;
f002 = tmp;
tmp = f021;
f021 = f022;
f022 = tmp;
tmp = f011;
f011 = f012;
f012 = tmp;
tmp = f201;
f201 = f202;
f202 = tmp;
tmp = f101;
f101 = f102;
f102 = tmp;
tmp = f221;
f221 = f222;
f222 = tmp;
tmp = f211;
f211 = f212;
f212 = tmp;
tmp = f121;
f121 = f122;
f122 = tmp;
tmp = f111;
f111 = f112;
f112 = tmp;

}

CudaDeviceFunction void EVelocity()
{
real_t Jx, Jy, Jz, rho;
rho  = ( f022 + f012 + f002 + f021 + f011 + f001 + f020 + f010 + f000 + ( f112 + f121 + f122 + f111 + f102 + f101 + f120 + f110 + f100 )*2. ) / ( 1 + Velocity );
Jx = Velocity*rho;
Jy  = ( -f022 + f012 - f021 + f011 - f020 + f010 ) / ( -1/3. );
Jz  = ( -f022 - f012 - f002 + f021 + f011 + f001 ) / ( -1/3. );
f200 = f100 - Jx*4./9.;
f210 = f120 + ( Jy - Jx )/9.;
f220 = f110 + ( -Jy - Jx )/9.;
f201 = f102 + ( Jz - Jx )/9.;
f211 = f122 + ( Jz + Jy - Jx )/36.;
f221 = f112 + ( Jz - Jy - Jx )/36.;
f202 = f101 + ( -Jz - Jx )/9.;
f212 = f121 + ( -Jz + Jy - Jx )/36.;
f222 = f111 + ( -Jz - Jy - Jx )/36.;

}

CudaDeviceFunction void WVelocity()
{
real_t Jx, Jy, Jz, rho;
rho  = ( f022 + f012 + f002 + f021 + f011 + f001 + f020 + f010 + f000 + ( f221 + f212 + f211 + f222 + f201 + f202 + f210 + f220 + f200 )*2. ) / ( 1 - Velocity );
Jx = Velocity*rho;
Jy  = ( -f022 + f012 - f021 + f011 - f020 + f010 ) / ( -1/3. );
Jz  = ( -f022 - f012 - f002 + f021 + f011 + f001 ) / ( -1/3. );
f100 = f200 + Jx*4./9.;
f110 = f220 + ( Jx + Jy )/9.;
f120 = f210 + ( -Jy + Jx )/9.;
f101 = f202 + ( Jz + Jx )/9.;
f111 = f222 + ( Jz + Jy + Jx )/36.;
f121 = f212 + ( Jz - Jy + Jx )/36.;
f102 = f201 + ( -Jz + Jx )/9.;
f112 = f221 + ( -Jz + Jy + Jx )/36.;
f122 = f211 + ( -Jz - Jy + Jx )/36.;

}

CudaDeviceFunction void SVelocity()
{
real_t Jx, Jy, Jz, rho;
rho  = ( f202 + f102 + f002 + f201 + f101 + f001 + f200 + f100 + f000 + ( f221 + f121 + f021 + f122 + f222 + f022 + f120 + f220 + f020 )*2. ) / ( 1 - Velocity );
Jy = Velocity*rho;
Jx  = ( -f202 + f102 - f201 + f101 - f200 + f100 ) / ( -1/3. );
Jz  = ( -f202 - f102 - f002 + f201 + f101 + f001 ) / ( -1/3. );
f010 = f020 + Jy*4./9.;
f110 = f220 + ( Jx + Jy )/9.;
f210 = f120 + ( Jy - Jx )/9.;
f011 = f022 + ( Jz + Jy )/9.;
f111 = f222 + ( Jz + Jy + Jx )/36.;
f211 = f122 + ( Jz + Jy - Jx )/36.;
f012 = f021 + ( -Jz + Jy )/9.;
f112 = f221 + ( -Jz + Jy + Jx )/36.;
f212 = f121 + ( -Jz + Jy - Jx )/36.;

}

CudaDeviceFunction void NVelocity()
{
real_t Jx, Jy, Jz, rho;
rho  = ( f202 + f102 + f002 + f201 + f101 + f001 + f200 + f100 + f000 + ( f112 + f212 + f012 + f211 + f111 + f011 + f210 + f110 + f010 )*2. ) / ( 1 + Velocity );
Jy = Velocity*rho;
Jx  = ( -f202 + f102 - f201 + f101 - f200 + f100 ) / ( -1/3. );
Jz  = ( -f202 - f102 - f002 + f201 + f101 + f001 ) / ( -1/3. );
f020 = f010 - Jy*4./9.;
f120 = f210 + ( -Jy + Jx )/9.;
f220 = f110 + ( -Jy - Jx )/9.;
f021 = f012 + ( Jz - Jy )/9.;
f121 = f212 + ( Jz - Jy + Jx )/36.;
f221 = f112 + ( Jz - Jy - Jx )/36.;
f022 = f011 + ( -Jz - Jy )/9.;
f122 = f211 + ( -Jz - Jy + Jx )/36.;
f222 = f111 + ( -Jz - Jy - Jx )/36.;

}

CudaDeviceFunction void WPressure()
{
real_t Jx, Jy, Jz, rho;
rho = 1 + Pressure*3.;
Jx  = ( f022 + f012 + f002 + f021 + f011 + f001 + f020 + f010 - rho + f000 + ( f221 + f212 + f211 + f222 + f201 + f202 + f210 + f220 + f200 )*2. ) / ( -1 );
Jy  = ( -f022 + f012 - f021 + f011 - f020 + f010 ) / ( -1/3. );
Jz  = ( -f022 - f012 - f002 + f021 + f011 + f001 ) / ( -1/3. );
f100 = f200 + Jx*4./9.;
f110 = f220 + ( Jx + Jy )/9.;
f120 = f210 + ( -Jy + Jx )/9.;
f101 = f202 + ( Jz + Jx )/9.;
f111 = f222 + ( Jz + Jy + Jx )/36.;
f121 = f212 + ( Jz - Jy + Jx )/36.;
f102 = f201 + ( -Jz + Jx )/9.;
f112 = f221 + ( -Jz + Jy + Jx )/36.;
f122 = f211 + ( -Jz - Jy + Jx )/36.;

}

CudaDeviceFunction void SPressure()
{
real_t Jx, Jy, Jz, rho;
rho = 1 + Pressure*3.;
Jy  = ( f202 + f102 + f002 + f201 + f101 + f001 - rho + f200 + f100 + f000 + ( f221 + f121 + f021 + f122 + f222 + f022 + f120 + f220 + f020 )*2. ) / ( -1 );
Jx  = ( -f202 + f102 - f201 + f101 - f200 + f100 ) / ( -1/3. );
Jz  = ( -f202 - f102 - f002 + f201 + f101 + f001 ) / ( -1/3. );
f010 = f020 + Jy*4./9.;
f110 = f220 + ( Jx + Jy )/9.;
f210 = f120 + ( Jy - Jx )/9.;
f011 = f022 + ( Jz + Jy )/9.;
f111 = f222 + ( Jz + Jy + Jx )/36.;
f211 = f122 + ( Jz + Jy - Jx )/36.;
f012 = f021 + ( -Jz + Jy )/9.;
f112 = f221 + ( -Jz + Jy + Jx )/36.;
f212 = f121 + ( -Jz + Jy - Jx )/36.;

}

CudaDeviceFunction void NPressure()
{
real_t Jx, Jy, Jz, rho;
rho = 1 + Pressure*3.;
Jy  = ( f202 + f102 + f002 + f201 + f101 + f001 - rho + f200 + f100 + f000 + ( f112 + f212 + f012 + f211 + f111 + f011 + f210 + f110 + f010 )*2. ) / ( 1 );
Jx  = ( -f202 + f102 - f201 + f101 - f200 + f100 ) / ( -1/3. );
Jz  = ( -f202 - f102 - f002 + f201 + f101 + f001 ) / ( -1/3. );
f020 = f010 - Jy*4./9.;
f120 = f210 + ( -Jy + Jx )/9.;
f220 = f110 + ( -Jy - Jx )/9.;
f021 = f012 + ( Jz - Jy )/9.;
f121 = f212 + ( Jz - Jy + Jx )/36.;
f221 = f112 + ( Jz - Jy - Jx )/36.;
f022 = f011 + ( -Jz - Jy )/9.;
f122 = f211 + ( -Jz - Jy + Jx )/36.;
f222 = f111 + ( -Jz - Jy - Jx )/36.;

}

CudaDeviceFunction void EPressure()
{
real_t Jx, Jy, Jz, rho;
rho = 1 + Pressure*3.;
Jx  = ( f022 + f012 + f002 + f021 + f011 + f001 + f020 + f010 - rho + f000 + ( f112 + f121 + f122 + f111 + f102 + f101 + f120 + f110 + f100 )*2. ) / ( 1 );
Jy  = ( -f022 + f012 - f021 + f011 - f020 + f010 ) / ( -1/3. );
Jz  = ( -f022 - f012 - f002 + f021 + f011 + f001 ) / ( -1/3. );
f200 = f100 - Jx*4./9.;
f210 = f120 + ( Jy - Jx )/9.;
f220 = f110 + ( -Jy - Jx )/9.;
f201 = f102 + ( Jz - Jx )/9.;
f211 = f122 + ( Jz + Jy - Jx )/36.;
f221 = f112 + ( Jz - Jy - Jx )/36.;
f202 = f101 + ( -Jz - Jx )/9.;
f212 = f121 + ( -Jz + Jy - Jx )/36.;
f222 = f111 + ( -Jz - Jy - Jx )/36.;

}
CudaDeviceFunction void TopSymmetry()
{
//Symmetry on the top of the boundary

f222 = f212;
f122 = f112;
f022 = f012;
f221 = f211;
f121 = f111;
f021 = f011;
f220 = f210;
f120 = f110;
f020 = f010;

}

CudaDeviceFunction void BottomSymmetry()
{
//Symmetry on the bottom of the boundary
f212=f222;
f112=f122;
f012=f022;
f211=f221;
f111=f121;
f011=f021;
f210=f220;
f110=f120;
f010=f020;

}

CudaDeviceFunction void Run() {
    switch (NodeType & NODE_BOUNDARY) {
	case NODE_TopSymmetry:
		TopSymmetry();
		break;
	case NODE_BottomSymmetry:
               	BottomSymmetry();
                break;
	case NODE_EPressure:
                EPressure();
               	break;
	case NODE_WPressure:
		WPressure();
		break;
	case NODE_SPressure:
                SPressure();
                break;
	case NODE_NPressure:
                NPressure();
                break;
	case NODE_WVelocity:
		WVelocity();
		break;
	 case NODE_NVelocity:
                NVelocity();
                break;
	 case NODE_SVelocity:
                SVelocity();
                break;
	case NODE_EVelocity:
		EVelocity();
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

f000 = ( rho*2. + ( -Jz*Jz - Jy*Jy - Jx*Jx )/rho*3. )*4./27.;
f100 = ( rho*2. + ( Jx*2. + ( -Jz*Jz - Jy*Jy + Jx*Jx*2. )/rho )*3. )/27.;
f200 = ( rho*2. + ( -Jx*2. + ( -Jz*Jz - Jy*Jy + Jx*Jx*2. )/rho )*3. )/27.;
f010 = ( rho*2. + ( Jy*2. + ( -Jz*Jz - Jx*Jx + Jy*Jy*2. )/rho )*3. )/27.;
f002 = ( rho*2. + ( -Jz*2. + ( -Jy*Jy - Jx*Jx + Jz*Jz*2. )/rho )*3. )/27.;
f001 = ( rho*2. + ( Jz*2. + ( -Jy*Jy - Jx*Jx + Jz*Jz*2. )/rho )*3. )/27.;
f020 = ( rho*2. + ( -Jy*2. + ( -Jz*Jz - Jx*Jx + Jy*Jy*2. )/rho )*3. )/27.;
	
f120 = rho*0.0185185185185185 + ( -Jz*Jz/rho + ( -Jy + Jx + ( Jy*Jy + ( Jx - Jy*3. )*Jx )/rho )*2. )/36.;
f220 = rho*0.0185185185185185 + ( -Jz*Jz/rho + ( -Jy - Jx + ( Jy*Jy + ( Jx + Jy*3. )*Jx )/rho )*2. )/36.;
f101 = rho*0.0185185185185185 + ( -Jy*Jy/rho + ( Jz + Jx + ( Jz*Jz + ( Jx + Jz*3. )*Jx )/rho )*2. )/36.;
f201 = rho*0.0185185185185185 + ( -Jy*Jy/rho + ( Jz - Jx + ( Jz*Jz + ( Jx - Jz*3. )*Jx )/rho )*2. )/36.;
f011 = rho*0.0185185185185185 + ( -Jx*Jx/rho + ( Jz + Jy + ( Jz*Jz + ( Jy + Jz*3. )*Jy )/rho )*2. )/36.;
f110 = rho*0.0185185185185185 + ( -Jz*Jz/rho + ( Jx + Jy + ( Jx*Jx + ( Jy + Jx*3. )*Jy )/rho )*2. )/36.;
f210 = rho*0.0185185185185185 + ( -Jz*Jz/rho + ( Jy - Jx + ( Jy*Jy + ( Jx - Jy*3. )*Jx )/rho )*2. )/36.;
f021 = rho*0.0185185185185185 + ( -Jx*Jx/rho + ( Jz - Jy + ( Jz*Jz + ( Jy - Jz*3. )*Jy )/rho )*2. )/36.;
f102 = rho*0.0185185185185185 + ( -Jy*Jy/rho + ( -Jz + Jx + ( Jz*Jz + ( Jx - Jz*3. )*Jx )/rho )*2. )/36.;
f202 = rho*0.0185185185185185 + ( -Jy*Jy/rho + ( -Jz - Jx + ( Jz*Jz + ( Jx + Jz*3. )*Jx )/rho )*2. )/36.;
f012 = rho*0.0185185185185185 + ( -Jx*Jx/rho + ( -Jz + Jy + ( Jz*Jz + ( Jy - Jz*3. )*Jy )/rho )*2. )/36.;
f022 = rho*0.0185185185185185 + ( -Jx*Jx/rho + ( -Jz - Jy + ( Jz*Jz + ( Jy + Jz*3. )*Jy )/rho )*2. )/36.;

f122 = ( rho + ( -Jz - Jy + Jx + ( Jz*Jz + Jy*Jy + Jx*Jx + ( Jz*Jy + ( -Jz - Jy )*Jx )*3. )/rho )*3. )*0.00462962962962963;
f222 = ( rho + ( -Jz - Jy - Jx + ( Jz*Jz + Jy*Jy + Jx*Jx + ( Jz*Jy + ( Jz + Jy )*Jx )*3. )/rho )*3. )*0.00462962962962963;
f121 = ( rho + ( Jz - Jy + Jx + ( Jz*Jz + Jy*Jy + Jx*Jx + ( -Jz*Jy + ( Jz - Jy )*Jx )*3. )/rho )*3. )*0.00462962962962963;
f221 = ( rho + ( Jz - Jy - Jx + ( Jz*Jz + Jy*Jy + Jx*Jx + ( -Jz*Jy + ( -Jz + Jy )*Jx )*3. )/rho )*3. )*0.00462962962962963;
f111 = ( rho + ( Jz + Jy + Jx + ( Jz*Jz + Jy*Jy + Jx*Jx + ( Jz*Jy + ( Jz + Jy )*Jx )*3. )/rho )*3. )*0.00462962962962963;
f211 = ( rho + ( Jz + Jy - Jx + ( Jz*Jz + Jy*Jy + Jx*Jx + ( Jz*Jy + ( -Jz - Jy )*Jx )*3. )/rho )*3. )*0.00462962962962963;
f112 = ( rho + ( -Jz + Jy + Jx + ( Jz*Jz + Jy*Jy + Jx*Jx + ( -Jz*Jy + ( -Jz + Jy )*Jx )*3. )/rho )*3. )*0.00462962962962963;
f212 = ( rho + ( -Jz + Jy - Jx + ( Jz*Jz + Jy*Jy + Jx*Jx + ( -Jz*Jy + ( Jz - Jy )*Jx )*3. )/rho )*3. )*0.00462962962962963;

}

CudaDeviceFunction void Init() {
	
	SetEquilibrum(1.0 + Pressure * 3.0, 0.0, 0.0, 0.0);
}

CudaDeviceFunction void CollisionMRT()
{	
	real_t omega= 1.0/(3*nu+0.5);
	 
 	real_t Jx, Jy, Jz;
	real_t rho;
	rho = f222 + f122 + f022 + f212 + f112 + f012 + f202 + f102 + f002 + f221 + f121 + f021 + f211 + f111 + f011 + f201 + f101 + f001 + f220 + f120 + f020 + f210 + f110 + f010 + f200 + f100 + f000;
	Jx   = -f222 + f122 - f212 + f112 - f202 + f102 - f221 + f121 - f211 + f111 - f201 + f101 - f220 + f120 - f210 + f110 - f200 + f100;
	Jy   = -f222 - f122 - f022 + f212 + f112 + f012 - f221 - f121 - f021 + f211 + f111 + f011 - f220 - f120 - f020 + f210 + f110 + f010;
	Jz   = -f222 - f122 - f022 - f212 - f112 - f012 - f202 - f102 - f002 + f221 + f121 + f021 + f211 + f111 + f011 + f201 + f101 + f001;
	
	f000 =(1.-omega)*f000 + omega*( ( rho*2. + ( -Jz*Jz - Jy*Jy - Jx*Jx )/rho*3. )*4./27.);
	f100 =(1.-omega)*f100 + omega*( ( rho*2. + ( Jx*2. + ( -Jz*Jz - Jy*Jy + Jx*Jx*2. )/rho )*3. )/27.);
	f200 =(1.-omega)*f200 + omega*( ( rho*2. + ( -Jx*2. + ( -Jz*Jz - Jy*Jy + Jx*Jx*2. )/rho )*3. )/27.);
	f010 =(1.-omega)*f010 + omega*( ( rho*2. + ( Jy*2. + ( -Jz*Jz - Jx*Jx + Jy*Jy*2. )/rho )*3. )/27.);
	f001 =(1.-omega)*f001 + omega*( ( rho*2. + ( Jz*2. + ( -Jy*Jy - Jx*Jx + Jz*Jz*2. )/rho )*3. )/27.);
	f020 =(1.-omega)*f020 + omega*( ( rho*2. + ( -Jy*2. + ( -Jz*Jz - Jx*Jx + Jy*Jy*2. )/rho )*3. )/27.);
	f002 =(1.-omega)*f002 + omega*( ( rho*2. + ( -Jz*2. + ( -Jy*Jy - Jx*Jx + Jz*Jz*2. )/rho )*3. )/27.);
	
	f110 =(1.-omega)*f110 + omega*( rho*0.0185185185185185 + ( -Jz*Jz/rho + (  Jx  + Jy + ( Jx*Jx + ( Jy + Jx*3. )*Jy )/rho )*2. )/36.);
	f210 =(1.-omega)*f210 + omega*( rho*0.0185185185185185 + ( -Jz*Jz/rho + (  Jy  -  Jx + ( Jy*Jy + ( Jx - Jy*3. )*Jx )/rho )*2. )/36.);
	f120 =(1.-omega)*f120 + omega*( rho*0.0185185185185185 + ( -Jz*Jz/rho + ( -Jy + Jx + ( Jy*Jy + ( Jx - Jy*3. )*Jx )/rho )*2. )/36.);
        f220 =(1.-omega)*f220 + omega*( rho*0.0185185185185185 + ( -Jz*Jz/rho + ( -Jy  - Jx + ( Jy*Jy + ( Jx + Jy*3. )*Jx )/rho )*2. )/36.);
        f101 =(1.-omega)*f101 + omega*( rho*0.0185185185185185 + ( -Jy*Jy/rho + (  Jz + Jx + ( Jz*Jz + ( Jx + Jz*3. )*Jx )/rho )*2. )/36.);
        f201 =(1.-omega)*f201 + omega*( rho*0.0185185185185185 + ( -Jy*Jy/rho + (  Jz  - Jx + ( Jz*Jz + ( Jx - Jz*3. )*Jx )/rho )*2. )/36.);
        f011 =(1.-omega)*f011 + omega*( rho*0.0185185185185185 + ( -Jx*Jx/rho + (  Jz + Jy + ( Jz*Jz + ( Jy + Jz*3. )*Jy )/rho )*2. )/36.);
	f022 =(1.-omega)*f022 + omega*( rho*0.0185185185185185 + ( -Jx*Jx/rho + ( -Jz  - Jy + ( Jz*Jz + ( Jy + Jz*3. )*Jy )/rho )*2. )/36.);
        f021 =(1.-omega)*f021 + omega*( rho*0.0185185185185185 + ( -Jx*Jx/rho + (  Jz  - Jy + ( Jz*Jz + ( Jy - Jz*3. )*Jy )/rho )*2. )/36.);
        f102 =(1.-omega)*f102 + omega*( rho*0.0185185185185185 + ( -Jy*Jy/rho + ( -Jz + Jx + ( Jz*Jz + ( Jx - Jz*3. )*Jx )/rho )*2. )/36.);
        f202 =(1.-omega)*f202 + omega*( rho*0.0185185185185185 + ( -Jy*Jy/rho + ( -Jz  - Jx + ( Jz*Jz + ( Jx + Jz*3. )*Jx )/rho )*2. )/36.);
        f012 =(1.-omega)*f012 + omega*( rho*0.0185185185185185 + ( -Jx*Jx/rho + ( -Jz + Jy + ( Jz*Jz + ( Jy - Jz*3. )*Jy )/rho )*2. )/36.);
	
        f112 =(1.-omega)*f112 + omega*( ( rho + ( -Jz + Jy + Jx + ( Jz*Jz + Jy*Jy + Jx*Jx + ( -Jz*Jy + ( -Jz + Jy )*Jx )*3. )/rho )*3. )*0.00462962962962963);
        f212 =(1.-omega)*f212 + omega*( ( rho + ( -Jz + Jy - Jx + ( Jz*Jz + Jy*Jy + Jx*Jx + ( -Jz*Jy + ( Jz - Jy )*Jx )*3. )/rho )*3. )*0.00462962962962963);
        f121 =(1.-omega)*f121 + omega*( ( rho + ( Jz - Jy + Jx + ( Jz*Jz + Jy*Jy + Jx*Jx + ( -Jz*Jy + ( Jz - Jy )*Jx )*3. )/rho )*3. )*0.00462962962962963);
        f221 =(1.-omega)*f221 + omega*( ( rho + ( Jz - Jy - Jx + ( Jz*Jz + Jy*Jy + Jx*Jx + ( -Jz*Jy + ( -Jz + Jy )*Jx )*3. )/rho )*3. )*0.00462962962962963);
	f111 =(1.-omega)*f111 + omega*( ( rho + ( Jz + Jy + Jx + ( Jz*Jz + Jy*Jy + Jx*Jx + ( Jz*Jy + ( Jz + Jy )*Jx )*3. )/rho )*3. )*0.00462962962962963);
        f211 =(1.-omega)*f211 + omega*( ( rho + ( Jz + Jy - Jx + ( Jz*Jz + Jy*Jy + Jx*Jx + ( Jz*Jy + ( -Jz - Jy )*Jx )*3. )/rho )*3. )*0.00462962962962963);
        f122 =(1.-omega)*f122 + omega*( ( rho + ( -Jz - Jy + Jx + ( Jz*Jz + Jy*Jy + Jx*Jx + ( Jz*Jy + ( -Jz - Jy )*Jx )*3. )/rho )*3. )*0.00462962962962963);
        f222 =(1.-omega)*f222 + omega*( ( rho + ( -Jz - Jy - Jx + ( Jz*Jz + Jy*Jy + Jx*Jx + ( Jz*Jy + ( Jz + Jy )*Jx )*3. )/rho )*3. )*0.00462962962962963);
	
	switch (NodeType & NODE_ADDITIONALS) {
		
		case NODE_XYslice1:
		AddToXYvx(Jx/rho);
		AddToXYvy(Jy/rho);
		AddToXYvz(Jz/rho);
		AddToXYrho1(rho);
		AddToXYarea(1);
		break;
		
		case NODE_XYslice2:
		AddToXYrho2(rho);
		break;
		
		case NODE_XZslice1:
		AddToXZvx(Jx/rho);
		AddToXZvy(Jy/rho);
		AddToXZvz(Jz/rho);
		AddToXZrho1(rho);
		AddToXZarea(1);
		break;
		
		case NODE_XZslice2:
		AddToXZrho2(rho);
		break;
		
		case NODE_YZslice1:
		AddToYZvx(Jx/rho);
		AddToYZvy(Jy/rho);
		AddToYZvz(Jz/rho);
		AddToYZrho1(rho);
		AddToYZarea(1);
		break;
		
		case NODE_YZslice2:
		AddToYZrho2(rho);
		break;
		
	}
	
	
	/* this conditional statement is either performed on the whole GPU or not because forces are defined globally */
	if( ForceX!=0.0 || ForceY  != 0.0 || ForceZ != 0.0 ){
		
	f000 -= ( rho*2. + ( -Jz*Jz - Jy*Jy - Jx*Jx )/rho*3. )*4./27.;
	f100 -= ( rho*2. + ( Jx*2. + ( -Jz*Jz - Jy*Jy + Jx*Jx*2. )/rho )*3. )/27.;
	f200 -= ( rho*2. + ( -Jx*2. + ( -Jz*Jz - Jy*Jy + Jx*Jx*2. )/rho )*3. )/27.;
	f010 -= ( rho*2. + ( Jy*2. + ( -Jz*Jz - Jx*Jx + Jy*Jy*2. )/rho )*3. )/27.;
	f002 -= ( rho*2. + ( -Jz*2. + ( -Jy*Jy - Jx*Jx + Jz*Jz*2. )/rho )*3. )/27.;
	f001 -= ( rho*2. + ( Jz*2. + ( -Jy*Jy - Jx*Jx + Jz*Jz*2. )/rho )*3. )/27.;
	f020 -= ( rho*2. + ( -Jy*2. + ( -Jz*Jz - Jx*Jx + Jy*Jy*2. )/rho )*3. )/27.;
	
	f120 -= rho*0.0185185185185185 + ( -Jz*Jz/rho + ( -Jy + Jx + ( Jy*Jy + ( Jx - Jy*3. )*Jx )/rho )*2. )/36.;
	f220 -= rho*0.0185185185185185 + ( -Jz*Jz/rho + ( -Jy - Jx + ( Jy*Jy + ( Jx + Jy*3. )*Jx )/rho )*2. )/36.;
	f101 -= rho*0.0185185185185185 + ( -Jy*Jy/rho + ( Jz + Jx + ( Jz*Jz + ( Jx + Jz*3. )*Jx )/rho )*2. )/36.;
	f201 -= rho*0.0185185185185185 + ( -Jy*Jy/rho + ( Jz - Jx + ( Jz*Jz + ( Jx - Jz*3. )*Jx )/rho )*2. )/36.;
	f011 -= rho*0.0185185185185185 + ( -Jx*Jx/rho + ( Jz + Jy + ( Jz*Jz + ( Jy + Jz*3. )*Jy )/rho )*2. )/36.;
	f110 -= rho*0.0185185185185185 + ( -Jz*Jz/rho + ( Jx + Jy + ( Jx*Jx + ( Jy + Jx*3. )*Jy )/rho )*2. )/36.;
	f210 -= rho*0.0185185185185185 + ( -Jz*Jz/rho + ( Jy - Jx + ( Jy*Jy + ( Jx - Jy*3. )*Jx )/rho )*2. )/36.;
	f021 -= rho*0.0185185185185185 + ( -Jx*Jx/rho + ( Jz - Jy + ( Jz*Jz + ( Jy - Jz*3. )*Jy )/rho )*2. )/36.;
	f102 -= rho*0.0185185185185185 + ( -Jy*Jy/rho + ( -Jz + Jx + ( Jz*Jz + ( Jx - Jz*3. )*Jx )/rho )*2. )/36.;
	f202 -= rho*0.0185185185185185 + ( -Jy*Jy/rho + ( -Jz - Jx + ( Jz*Jz + ( Jx + Jz*3. )*Jx )/rho )*2. )/36.;
	f012 -= rho*0.0185185185185185 + ( -Jx*Jx/rho + ( -Jz + Jy + ( Jz*Jz + ( Jy - Jz*3. )*Jy )/rho )*2. )/36.;
	f022 -= rho*0.0185185185185185 + ( -Jx*Jx/rho + ( -Jz - Jy + ( Jz*Jz + ( Jy + Jz*3. )*Jy )/rho )*2. )/36.;
	
	f112 -= ( rho + ( -Jz + Jy + Jx + ( Jz*Jz + Jy*Jy + Jx*Jx + ( -Jz*Jy + ( -Jz + Jy )*Jx )*3. )/rho )*3. )*0.00462962962962963;
	f212 -= ( rho + ( -Jz + Jy - Jx + ( Jz*Jz + Jy*Jy + Jx*Jx + ( -Jz*Jy + ( Jz - Jy )*Jx )*3. )/rho )*3. )*0.00462962962962963;
	f121 -= ( rho + ( Jz - Jy + Jx + ( Jz*Jz + Jy*Jy + Jx*Jx + ( -Jz*Jy + ( Jz - Jy )*Jx )*3. )/rho )*3. )*0.00462962962962963;
	f221 -= ( rho + ( Jz - Jy - Jx + ( Jz*Jz + Jy*Jy + Jx*Jx + ( -Jz*Jy + ( -Jz + Jy )*Jx )*3. )/rho )*3. )*0.00462962962962963;
	f122 -= ( rho + ( -Jz - Jy + Jx + ( Jz*Jz + Jy*Jy + Jx*Jx + ( Jz*Jy + ( -Jz - Jy )*Jx )*3. )/rho )*3. )*0.00462962962962963;
	f222 -= ( rho + ( -Jz - Jy - Jx + ( Jz*Jz + Jy*Jy + Jx*Jx + ( Jz*Jy + ( Jz + Jy )*Jx )*3. )/rho )*3. )*0.00462962962962963;
	f111 -= ( rho + ( Jz + Jy + Jx + ( Jz*Jz + Jy*Jy + Jx*Jx + ( Jz*Jy + ( Jz + Jy )*Jx )*3. )/rho )*3. )*0.00462962962962963;
	f211 -= ( rho + ( Jz + Jy - Jx + ( Jz*Jz + Jy*Jy + Jx*Jx + ( Jz*Jy + ( -Jz - Jy )*Jx )*3. )/rho )*3. )*0.00462962962962963;	
	Jx+= ForceX;
	Jy+= ForceY;
	Jz+= ForceZ;

	f000 += ( rho*2. + ( -Jz*Jz - Jy*Jy - Jx*Jx )/rho*3. )*4./27.;
	f100 += ( rho*2. + ( Jx*2. + ( -Jz*Jz - Jy*Jy + Jx*Jx*2. )/rho )*3. )/27.;
	f200 += ( rho*2. + ( -Jx*2. + ( -Jz*Jz - Jy*Jy + Jx*Jx*2. )/rho )*3. )/27.;
	f010 += ( rho*2. + ( Jy*2. + ( -Jz*Jz - Jx*Jx + Jy*Jy*2. )/rho )*3. )/27.;
	f001 += ( rho*2. + ( Jz*2. + ( -Jy*Jy - Jx*Jx + Jz*Jz*2. )/rho )*3. )/27.;
	f002 += ( rho*2. + ( -Jz*2. + ( -Jy*Jy - Jx*Jx + Jz*Jz*2. )/rho )*3. )/27.;
	f020 += ( rho*2. + ( -Jy*2. + ( -Jz*Jz - Jx*Jx + Jy*Jy*2. )/rho )*3. )/27.;
	
	f110 += rho*0.0185185185185185 + ( -Jz*Jz/rho + ( Jx + Jy + ( Jx*Jx + ( Jy + Jx*3. )*Jy )/rho )*2. )/36.;
	f210 += rho*0.0185185185185185 + ( -Jz*Jz/rho + ( Jy - Jx + ( Jy*Jy + ( Jx - Jy*3. )*Jx )/rho )*2. )/36.;
	f120 += rho*0.0185185185185185 + ( -Jz*Jz/rho + ( -Jy + Jx + ( Jy*Jy + ( Jx - Jy*3. )*Jx )/rho )*2. )/36.;
	f220 += rho*0.0185185185185185 + ( -Jz*Jz/rho + ( -Jy - Jx + ( Jy*Jy + ( Jx + Jy*3. )*Jx )/rho )*2. )/36.;
	f101 += rho*0.0185185185185185 + ( -Jy*Jy/rho + ( Jz + Jx + ( Jz*Jz + ( Jx + Jz*3. )*Jx )/rho )*2. )/36.;
	f201 += rho*0.0185185185185185 + ( -Jy*Jy/rho + ( Jz - Jx + ( Jz*Jz + ( Jx - Jz*3. )*Jx )/rho )*2. )/36.;
	f102 += rho*0.0185185185185185 + ( -Jy*Jy/rho + ( -Jz + Jx + ( Jz*Jz + ( Jx - Jz*3. )*Jx )/rho )*2. )/36.;
	f202 += rho*0.0185185185185185 + ( -Jy*Jy/rho + ( -Jz - Jx + ( Jz*Jz + ( Jx + Jz*3. )*Jx )/rho )*2. )/36.;
	f012 += rho*0.0185185185185185 + ( -Jx*Jx/rho + ( -Jz + Jy + ( Jz*Jz + ( Jy - Jz*3. )*Jy )/rho )*2. )/36.;
	f011 += rho*0.0185185185185185 + ( -Jx*Jx/rho + ( Jz + Jy + ( Jz*Jz + ( Jy + Jz*3. )*Jy )/rho )*2. )/36.;
	f021 += rho*0.0185185185185185 + ( -Jx*Jx/rho + ( Jz - Jy + ( Jz*Jz + ( Jy - Jz*3. )*Jy )/rho )*2. )/36.;
	f022 += rho*0.0185185185185185 + ( -Jx*Jx/rho + ( -Jz - Jy + ( Jz*Jz + ( Jy + Jz*3. )*Jy )/rho )*2. )/36.;
	
	f121 += ( rho + (  Jz - Jy + Jx + ( Jz*Jz + Jy*Jy + Jx*Jx + ( -Jz*Jy + ( Jz - Jy )*Jx )*3. )/rho )*3. )*0.00462962962962963;
	f221 += ( rho + (  Jz - Jy - Jx + ( Jz*Jz + Jy*Jy + Jx*Jx + ( -Jz*Jy + ( -Jz + Jy )*Jx )*3. )/rho )*3. )*0.00462962962962963;
	f111 += ( rho + (  Jz + Jy + Jx + ( Jz*Jz + Jy*Jy + Jx*Jx + ( Jz*Jy + ( Jz + Jy )*Jx )*3. )/rho )*3. )*0.00462962962962963;
	f211 += ( rho + (  Jz + Jy - Jx + ( Jz*Jz + Jy*Jy + Jx*Jx + ( Jz*Jy + ( -Jz - Jy )*Jx )*3. )/rho )*3. )*0.00462962962962963;
	f112 += ( rho + ( -Jz + Jy + Jx + ( Jz*Jz + Jy*Jy + Jx*Jx + ( -Jz*Jy + ( -Jz + Jy )*Jx )*3. )/rho )*3. )*0.00462962962962963;
	f212 += ( rho + ( -Jz + Jy - Jx + ( Jz*Jz + Jy*Jy + Jx*Jx + ( -Jz*Jy + ( Jz - Jy )*Jx )*3. )/rho )*3. )*0.00462962962962963;
	f122 += ( rho + ( -Jz - Jy + Jx + ( Jz*Jz + Jy*Jy + Jx*Jx + ( Jz*Jy + ( -Jz - Jy )*Jx )*3. )/rho )*3. )*0.00462962962962963;
	f222 += ( rho + ( -Jz - Jy - Jx + ( Jz*Jz + Jy*Jy + Jx*Jx + ( Jz*Jy + ( Jz + Jy )*Jx )*3. )/rho )*3. )*0.00462962962962963;
		
	}
	
	
}

