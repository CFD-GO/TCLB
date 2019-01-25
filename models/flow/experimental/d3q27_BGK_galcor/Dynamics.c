/*-------------------------------------------------------------*/
/*  	CLB - Cudne LB - Stencil Version                           */
/*     CUDA based Adjoint Lattice Boltzmann Solver             */
/*     Author: Wojciech Regulski                          */
/*     Author e-mail wojtek.regulski@gmail.com
/*     Developed at: Warsaw University of Technology - 2012    */
/*
	Description:
	 - collision file implements BGK model with Galilean correction [1]
	 - force term implemented in the Kuperstokh way [2] 
	 - collision file can additionally drop data to buffers defines as XY_Slice1, XY_Slice2...

	Model description:
	This is the NEW implementation of the BGK collision model with the Gallilean-correctoin present in the equilibrium distribution [1].
	The equilibrium has the PRODUCT_FORM now and the model has a significantly improved stability over standard BGK.
	
	f_eq_ijk = - rho * X_i*Y_j*Z_k for		 i,j,k=-1,0,1
	
	with each of the elements has a Galilean correction in it, Gx,Gy,Gz. Galilean correction required 2nd moments of the distribution functions, M2x, M2y, M2z:
	
	2nd moments of distribution functions:
	M2x=	+f100+f200+f110+f210+f220+f120+f101+f201+f102+f202+f111+f211+f221+f121+f112+f212+f222+f122;
	M2y=	+f010+f020+f110+f210+f220+f120+f011+f021+f012+f022+f111+f211+f221+f121+f112+f212+f222+f122;
	M2z=	+f001+f002+f101+f011+f201+f021+f102+f012+f202+f022+f111+f211+f221+f121+f112+f212+f222+f122;
	
	velocity derivatives:
	DxUx = -omega*(1.5*M2x*RhoInv-0.5-1.5*Ux*Ux); 	
	DyUy = -omega*(1.5*M2y*RhoInv-0.5-1.5*Uy*Uy);	
	DzUz = -omega*(1.5*M2z*RhoInv-0.5-1.5*Uz*Uz);

	Gallilean corrections:
	Gx = -9.*Ux*Ux * DxUx * nu;	
	Gy = -9.*Uy*Uy * DyUy * nu;	
	Gz = -9.*Uz*Uz * DzUz * nu;

	product elements:
	X_0 = -2./3. + Ux*Ux + Gx;		Y_0 = -2./3. + Uy*Uy + Gy;		Z_0 = -2./3. + Uz*Uz + Gz;
	X_1 = -0.5*(X_0 + 1. + Ux);		Y_1 = -0.5*(Y_0 + 1. + Uy);		Z_1 = -0.5*(Z_0 + 1. + Uz);
	X_2 = X_1 + Ux;				Y_2 = Y_1 + Uy;				Z_2 = Z_1 + Uz;
	
	Auxuliary variables:
	RhoInv = 1./rho 	 - inverse density
	omega  = 1/tau 	 - inverse of relaxation time
	nu = 1./3/*(tau-0.5)  - kinematic viscosity
	
	References:
	[1]   Geier M., Schönherr M., Pasquali A., Krafczyk M., The cumulant lattice Boltzmann equation in three dimensions: Theory and validation. Comput. Math. Appl. , 2015, 70, 507
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
real_t tmp;f222 + f122 + f022 + f212 + f112 + f012 + f202 + f102 + f002 + f221 + f121 + f021 + f211 + f111 + f011 + f201 + f101 + f001 + f220 + f120 + f020 + f210 + f110 + f010 + f200 + f100 + f000;
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
	
	/* equilibrium distribution in the product form - see file header and references therein for detailed explanation */
	real_t X_0, X_1, X_2;
	real_t Y_0, Y_1, Y_2;
	real_t Z_0, Z_1, Z_2;
	real_t Ux, Uy, Uz;
	real_t Gx, Gy, Gz; //  Galilean corrections - they are usually non zero, but they are left here for code consistency with the collision term
	Gx = 0.0;		Gy = 0.0;		Gz = 0.0;
	Ux = Jx/rho; 	Uy = Jy/rho; 	Uz = Jz/rho;
	
	
	/* product elements */
	X_0 = -2./3. + Ux*Ux + Gx;		Y_0 = -2./3. + Uy*Uy + Gy;		Z_0 = -2./3. + Uz*Uz + Gz;
	X_1 = -0.5*(X_0 + 1. + Ux);		Y_1 = -0.5*(Y_0 + 1. + Uy);		Z_1 = -0.5*(Z_0 + 1. + Uz);
	X_2 = X_1 + Ux;				Y_2 = Y_1 + Uy;				Z_2 = Z_1 + Uz;
	
	f000=-rho*X_0*Y_0*Z_0;
	f100=-rho*X_1*Y_0*Z_0;
	f010=-rho*X_0*Y_1*Z_0;
	f200=-rho*X_2*Y_0*Z_0;
	f020=-rho*X_0*Y_2*Z_0;
	f001=-rho*X_0*Y_0*Z_1;
	f002=-rho*X_0*Y_0*Z_2;

	f110=-rho*X_1*Y_1*Z_0;
	f210=-rho*X_2*Y_1*Z_0;
	f220=-rho*X_2*Y_2*Z_0;
	f120=-rho*X_1*Y_2*Z_0;
	f101=-rho*X_1*Y_0*Z_1;
	f011=-rho*X_0*Y_1*Z_1;
	f201=-rho*X_2*Y_0*Z_1;
	f021=-rho*X_0*Y_2*Z_1;
	f102=-rho*X_1*Y_0*Z_2;
	f012=-rho*X_0*Y_1*Z_2;
	f202=-rho*X_2*Y_0*Z_2;
	f022=-rho*X_0*Y_2*Z_2;

	f111=-rho*X_1*Y_1*Z_1;
	f211=-rho*X_2*Y_1*Z_1;
	f221=-rho*X_2*Y_2*Z_1;
	f121=-rho*X_1*Y_2*Z_1;
	f112=-rho*X_1*Y_1*Z_2;
	f212=-rho*X_2*Y_1*Z_2;
	f222=-rho*X_2*Y_2*Z_2;
	f122=-rho*X_1*Y_2*Z_2;

}

CudaDeviceFunction void Init() {
	
	SetEquilibrum(1.0 + Pressure * 3.0, 0.0, 0.0, 0.0);
}

CudaDeviceFunction void CollisionMRT()
{	
	real_t omega= 1.0/(3*nu+0.5);
	 
 	real_t Ux, Uy, Uz;
	real_t rho;
	
	rho = f222 + f122 + f022 + f212 + f112 + f012 + f202 + f102 + f002 + f221 + f121 + f021 + f211 + f111 + f011 + f201 + f101 + f001 + f220 + f120 + f020 + f210 + f110 + f010 + f200 + f100 + f000;
	
	real_t rhoInv = 1./rho;
	
	real_t X_0, X_1, X_2;
	real_t Y_0, Y_1, Y_2;
	real_t Z_0, Z_1, Z_2;
	real_t Gx, Gy, Gz;		// Galilean correciton terms
	real_t DxUx, DyUy, DzUz; 	// velocity derivatives
	
	Ux = -f222 + f122 - f212 + f112 - f202 + f102 - f221 + f121 - f211 + f111 - f201 + f101 - f220 + f120 - f210 + f110 - f200 + f100;
	Uy = -f222 - f122 - f022 + f212 + f112 + f012 - f221 - f121 - f021 + f211 + f111 + f011 - f220 - f120 - f020 + f210 + f110 + f010;
	Uz = -f222 - f122 - f022 - f212 - f112 - f012 - f202 - f102 - f002 + f221 + f121 + f021 + f211 + f111 + f011 + f201 + f101 + f001;
	
	Ux *= rhoInv;
	Uy *= rhoInv;
	Uz *= rhoInv;
	
	/* 2nd moments of distribution function */
	
	real_t M2x=	+f100+f200+f110+f210+f220+f120+f101+f201+f102+f202+f111+f211+f221+f121+f112+f212+f222+f122;
	real_t M2y=	+f010+f020+f110+f210+f220+f120+f011+f021+f012+f022+f111+f211+f221+f121+f112+f212+f222+f122;
	real_t M2z=	+f001+f002+f101+f011+f201+f021+f102+f012+f202+f022+f111+f211+f221+f121+f112+f212+f222+f122;
	
	/* velocity derivatives */
	DxUx = -omega*(1.5*M2x*rhoInv-0.5-1.5*Ux*Ux); 	DyUy = -omega*(1.5*M2y*rhoInv-0.5-1.5*Uy*Uy);	DzUz = -omega*(1.5*M2z*rhoInv-0.5-1.5*Uz*Uz);

	/* Gallilean corrections */
	Gx = -9.*Ux*Ux * DxUx * nu;	Gy = -9.*Uy*Uy * DyUy * nu;	Gz = -9.*Uz*Uz * DzUz * nu;

	/* product elements */
	X_0 = -2./3. + Ux*Ux + Gx;		Y_0 = -2./3. + Uy*Uy + Gy;		Z_0 = -2./3. + Uz*Uz + Gz;
	X_1 = -0.5*(X_0 + 1. + Ux);		Y_1 = -0.5*(Y_0 + 1. + Uy);		Z_1 = -0.5*(Z_0 + 1. + Uz);
	X_2 = X_1 + Ux;				Y_2 = Y_1 + Uy;				Z_2 = Z_1 + Uz;
	
	f000=(1.-omega)*f000-omega*rho*X_0*Y_0*Z_0;
	f100=(1.-omega)*f100-omega*rho*X_1*Y_0*Z_0;
	f010=(1.-omega)*f010-omega*rho*X_0*Y_1*Z_0;
	f200=(1.-omega)*f200-omega*rho*X_2*Y_0*Z_0;
	f020=(1.-omega)*f020-omega*rho*X_0*Y_2*Z_0;
	f001=(1.-omega)*f001-omega*rho*X_0*Y_0*Z_1;
	f002=(1.-omega)*f002-omega*rho*X_0*Y_0*Z_2;

	f110=(1.-omega)*f110-omega*rho*X_1*Y_1*Z_0;
	f210=(1.-omega)*f210-omega*rho*X_2*Y_1*Z_0;
	f220=(1.-omega)*f220-omega*rho*X_2*Y_2*Z_0;
	f120=(1.-omega)*f120-omega*rho*X_1*Y_2*Z_0;
	f101=(1.-omega)*f101-omega*rho*X_1*Y_0*Z_1;
	f011=(1.-omega)*f011-omega*rho*X_0*Y_1*Z_1;
	f201=(1.-omega)*f201-omega*rho*X_2*Y_0*Z_1;
	f021=(1.-omega)*f021-omega*rho*X_0*Y_2*Z_1;
	f102=(1.-omega)*f102-omega*rho*X_1*Y_0*Z_2;
	f012=(1.-omega)*f012-omega*rho*X_0*Y_1*Z_2;
	f202=(1.-omega)*f202-omega*rho*X_2*Y_0*Z_2;
	f022=(1.-omega)*f022-omega*rho*X_0*Y_2*Z_2;

	f111=(1.-omega)*f111-omega*rho*X_1*Y_1*Z_1;
	f211=(1.-omega)*f211-omega*rho*X_2*Y_1*Z_1;
	f221=(1.-omega)*f221-omega*rho*X_2*Y_2*Z_1;
	f121=(1.-omega)*f121-omega*rho*X_1*Y_2*Z_1;
	f112=(1.-omega)*f112-omega*rho*X_1*Y_1*Z_2;
	f212=(1.-omega)*f212-omega*rho*X_2*Y_1*Z_2;
	f222=(1.-omega)*f222-omega*rho*X_2*Y_2*Z_2;
	f122=(1.-omega)*f122-omega*rho*X_1*Y_2*Z_2;
	
	/* this conditional statement is either performed on the whole GPU or not because forces are defined globally */
	if( ForceX!=0.0 || ForceY  != 0.0 || ForceZ != 0.0 ){
		
		/* add Force term in Kuperstokh way - see top of the file for description */
		/* 1. We subtract the previous equilibrium. Of course this comes with "+" because equilibria have NEGATIVE signs */
		
	f000+=rho*X_0*Y_0*Z_0;		
	f100+=rho*X_1*Y_0*Z_0;		f010+=rho*X_0*Y_1*Z_0;		f001+=rho*X_0*Y_0*Z_1;
	f200+=rho*X_2*Y_0*Z_0;		f020+=rho*X_0*Y_2*Z_0;		f002+=rho*X_0*Y_0*Z_2;

	f110+=rho*X_1*Y_1*Z_0;		f210+=rho*X_2*Y_1*Z_0;		f220+=rho*X_2*Y_2*Z_0;
	f120+=rho*X_1*Y_2*Z_0;		f101+=rho*X_1*Y_0*Z_1;		f011+=rho*X_0*Y_1*Z_1;
	f201+=rho*X_2*Y_0*Z_1;		f021+=rho*X_0*Y_2*Z_1;		f102+=rho*X_1*Y_0*Z_2;
	f012+=rho*X_0*Y_1*Z_2;		f202+=rho*X_2*Y_0*Z_2;		f022+=rho*X_0*Y_2*Z_2;

	f111+=rho*X_1*Y_1*Z_1;		f211+=rho*X_2*Y_1*Z_1;		f221+=rho*X_2*Y_2*Z_1;
	f121+=rho*X_1*Y_2*Z_1;		f112+=rho*X_1*Y_1*Z_2;		f212+=rho*X_2*Y_1*Z_2;
	f222+=rho*X_2*Y_2*Z_2;		f122+=rho*X_1*Y_2*Z_2;
		
		/*2. Velocity modification, new Gallilean corrections and new products
		NOTE: We do not modify DxUx and other derivatives - for spatially uniform force field that is not a big problem because pressure differences are not big and local force contributions do not differ much */
	
	Ux+= ForceX/rho;	Uy+= ForceY/rho;		Uz+= ForceZ/rho;
	Gx = -9.*Ux*Ux * DxUx * nu;	Gy = -9.*Uy*Uy * DyUy * nu;	Gz = -9.*Uz*Uz * DzUz * nu;

	X_0 = -2./3. + Ux*Ux + Gx;		Y_0 = -2./3. + Uy*Uy + Gy;		Z_0 = -2./3. + Uz*Uz + Gz;
	X_1 = -0.5*(X_0 + 1. + Ux);		Y_1 = -0.5*(Y_0 + 1. + Uy);		Z_1 = -0.5*(Z_0 + 1. + Uz);
	X_2 = X_1 + Ux;				Y_2 = Y_1 + Uy;				Z_2 = Z_1 + Uz;

		/* 3. Add new equilibria to distributions */

	f000-=rho*X_0*Y_0*Z_0;
	f100-=rho*X_1*Y_0*Z_0;
	f010-=rho*X_0*Y_1*Z_0;
	f200-=rho*X_2*Y_0*Z_0;
	f020-=rho*X_0*Y_2*Z_0;
	f001-=rho*X_0*Y_0*Z_1;
	f002-=rho*X_0*Y_0*Z_2;

	f110-=rho*X_1*Y_1*Z_0;
	f210-=rho*X_2*Y_1*Z_0;
	f220-=rho*X_2*Y_2*Z_0;
	f120-=rho*X_1*Y_2*Z_0;
	f101-=rho*X_1*Y_0*Z_1;
	f011-=rho*X_0*Y_1*Z_1;
	f201-=rho*X_2*Y_0*Z_1;
	f021-=rho*X_0*Y_2*Z_1;
	f102-=rho*X_1*Y_0*Z_2;
	f012-=rho*X_0*Y_1*Z_2;
	f202-=rho*X_2*Y_0*Z_2;
	f022-=rho*X_0*Y_2*Z_2;

	f111-=rho*X_1*Y_1*Z_1;
	f211-=rho*X_2*Y_1*Z_1;
	f221-=rho*X_2*Y_2*Z_1;
	f121-=rho*X_1*Y_2*Z_1;
	f112-=rho*X_1*Y_1*Z_2;
	f212-=rho*X_2*Y_1*Z_2;
	f222-=rho*X_2*Y_2*Z_2;
	f122-=rho*X_1*Y_2*Z_2;
		
	}
	
	/* adding stuff to lacal buffers */
	switch (NodeType & NODE_ADDITIONALS) {
		
		case NODE_XYslice1:
		AddToXYvx(Ux+0.5*ForceX);
		AddToXYvy(Uy+0.5*ForceY);
		AddToXYvz(Uz+0.5*ForceZ);
		AddToXYrho1(rho);
		AddToXYarea(1);
		break;
		
		case NODE_XYslice2:
		AddToXYrho2(rho);
		break;
		
		case NODE_XZslice1:
		AddToXZvx(Ux+0.5*ForceX);
		AddToXZvy(Uy+0.5*ForceY);
		AddToXZvz(Uz+0.5*ForceZ);
		AddToXZrho1(rho);
		AddToXZarea(1);
		break;
		
		case NODE_XZslice2:
		AddToXZrho2(rho);
		break;
		
		case NODE_YZslice1:
		AddToYZvx(Ux+0.5*ForceX);
		AddToYZvy(Uy+0.5*ForceY);
		AddToYZvz(Uz+0.5*ForceZ);
		AddToYZrho1(rho);
		AddToYZarea(1);
		break;
		
		case NODE_YZslice2:
		AddToYZrho2(rho);
		break;
		
	}
}

