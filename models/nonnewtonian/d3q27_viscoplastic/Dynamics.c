/*-------------------------------------------------------------*/
/*  	CLB - Cudne LB - Stencil Version                       	    */
/*     CUDA based Adjoint Lattice Boltzmann Solver        	     */
/*     Author: Wojciech Regulski                          			*/
/*     Author e-mail wregulski@meil.pw.edu.pl
/*     Developed at: Warsaw University of Technology - 2012-2016    */
/*
	Description:
	This is an implementation of a very clever viscoplastic fluid model described in this paper:

	Vikhansky, A. Lattice-Boltzmann method for yield-stress liquids. Journal of Non-Newtonian Fluid Mechanics, 155(3):95 - 100, 2008.
	
	More details on this model will be availabe in:
	
	W. Regulski, C. Leonardi, J. Szumbarski, On the spatial convergence and transient behavior of lattice Boltzmann methods for modeling fluids with yield stress
	
	As soon as the latter paper is published, an exact reference will be provided.

/*-------------------------------------------------------------*/


CudaDeviceFunction real_t getRho(){
	return f222 + f122 + f022 + f212 + f112 + f012 + f202 + f102 + f002 + f221 + f121 + f021 + f211 + f111 + f011 + f201 + f101 + f001 + f220 + f120 + f020 + f210 + f110 + f010 + f200 + f100 + f000;
}

CudaDeviceFunction real_t getP(){
	return ((f222 + f122 + f022 + f212 + f112 + f012 + f202 + f102 + f002 + f221 + f121 + f021 + f211 + f111 + f011 + f201 + f101 + f001 + f220 + f120 + f020 + f210 + f110 + f010 + f200 + f100 + f000)-1.0)/3.0;
}

CudaDeviceFunction real_t getnu_app(){
	return nu_app;
	}

CudaDeviceFunction real_t getyield_stat(){
	return yield_stat;
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
f120 = tmp;
	
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

CudaDeviceFunction void EVelocity_ZouHe()
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

CudaDeviceFunction void WVelocity_ZouHe()
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

CudaDeviceFunction void SVelocity_ZouHe()
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

CudaDeviceFunction void NVelocity_ZouHe()
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

CudaDeviceFunction void EPressure_ZouHe()
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

CudaDeviceFunction void WPressure_ZouHe()
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

CudaDeviceFunction void SPressure_ZouHe()
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

CudaDeviceFunction void NPressure_ZouHe()
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


CudaDeviceFunction void Run() {
    switch (NodeType & NODE_BOUNDARY) {
	case NODE_EPressure_ZouHe:
                EPressure_ZouHe();
               	break;
	case NODE_WPressure_ZouHe:
		WPressure_ZouHe();
		break;
	case NODE_SPressure_ZouHe:
                SPressure_ZouHe();
                break;
	case NODE_NPressure_ZouHe:
                NPressure_ZouHe();
                break;
	case NODE_WVelocity_ZouHe:
		WVelocity_ZouHe();
		break;
	 case NODE_NVelocity_ZouHe:
                NVelocity_ZouHe();
                break;
	 case NODE_SVelocity_ZouHe:
                SVelocity_ZouHe();
                break;
	case NODE_EVelocity_ZouHe:
		EVelocity_ZouHe();
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


CudaDeviceFunction void Init(){
	
	nu_app = 0.0;
	yield_stat = 0.0;
	SetEquilibrum(1.0 + Pressure * 3.0, 0.0, 0.0, 0.0);
}

CudaDeviceFunction void CollisionMRT()
{
	
	real_t Rho =  f222 + f122 + f022 + f212 + f112 + f012 + f202 + f102 + f002 + f221 + f121 + f021 + f211 + f111 + f011 + f201 + f101 + f001 + f220 + f120 + f020 + f210 + f110 + f010 + f200 + f100 + f000;
	real_t RhoInv = 1./Rho;
	
	/* extra variables that carry force - specific implementation for this model */
	
	real_t Phi000;
	real_t Phi100	, Phi010	, Phi200	, Phi020	, Phi001	, Phi002;
	real_t Phi110	, Phi210	, Phi220	, Phi120	, Phi101	, Phi011	, Phi201	, Phi021	, Phi102	, Phi012	, Phi202	, Phi022;
	real_t Phi111	, Phi211	, Phi221	, Phi121	, Phi112	, Phi212	, Phi222	, Phi122;
	
	/* equilibrium populations */

	real_t f_eq000;
	real_t f_eq100	, f_eq010	, f_eq200	, f_eq020	, f_eq001	, f_eq002;
	real_t f_eq110	, f_eq210	, f_eq220	, f_eq120	, f_eq101	, f_eq011	, f_eq201	, f_eq021	, f_eq102	, f_eq012	, f_eq202	, f_eq022;
	real_t f_eq111	, f_eq211	, f_eq221	, f_eq121	, f_eq112	, f_eq212	, f_eq222	, f_eq122;
	
	real_t Ux = (-f222 + f122 - f212 + f112 - f202 + f102 - f221 + f121 - f211 + f111 - f201 + f101 - f220 + f120 - f210 + f110 - f200 + f100)*RhoInv + ForceX*0.5;
	real_t Uy = (-f222 - f122 - f022 + f212 + f112 + f012 - f221 - f121 - f021 + f211 + f111 + f011 - f220 - f120 - f020 + f210 + f110 + f010)*RhoInv + ForceY*0.5;
	real_t Uz = (-f222 - f122 - f022 - f212 - f112 - f012 - f202 - f102 - f002 + f221 + f121 + f021 + f211 + f111 + f011 + f201 + f101 + f001)*RhoInv + ForceZ*0.5;
	real_t Usq = Ux*Ux+Uy*Uy+Uz*Uz;

	/* partial Rho - stores consecutive fractions of density */
	real_t rho_par = Rho * 2./9.;
	Phi000 =	0.0;				
	Phi100 =	rho_par*( ForceX);	Phi010 =	rho_par*(+ForceY);		Phi200 =	rho_par*(-ForceX);
	Phi020 =	rho_par*(-ForceY);	Phi001 =	rho_par*(+ForceZ);		Phi002 =	rho_par*(-ForceZ);
	rho_par *=0.25;
	Phi110 =	rho_par*(ForceX+ForceY);	Phi210 =rho_par*( -ForceX+ForceY);	Phi220 =	rho_par*(-ForceX-ForceY);	Phi120=	rho_par*( ForceX -ForceY);
	Phi101=	rho_par*(ForceX+ForceZ);	Phi011 =rho_par*(+ForceY+ForceZ);	Phi201=	rho_par*(-ForceX+ForceZ);	Phi021=	rho_par*(-ForceY+ForceZ);
	Phi102=	rho_par*(ForceX-ForceZ);		Phi012 =rho_par*(+ForceY -ForceZ);	Phi202=	rho_par*(-ForceX-ForceZ);	Phi022=	rho_par*(-ForceY -ForceZ);
	rho_par *=0.25;
	Phi111=	rho_par*(ForceX+ForceY+ForceZ);	Phi211=rho_par*(-ForceX+ForceY+ForceZ);	Phi221=rho_par*(-ForceX-ForceY+ForceZ);	Phi121=	rho_par*(ForceX-ForceY+ForceZ);
	Phi112=	rho_par*(ForceX+ForceY-ForceZ);	Phi212=rho_par*(-ForceX+ForceY-ForceZ);	Phi222=rho_par*(-ForceX-ForceY-ForceZ);	Phi122=	rho_par*(ForceX-ForceY-ForceZ);
	
	/* modify the equilibria by subtracting half-Phi */

	rho_par = Rho * 8./27.;
	f_eq000  = rho_par*( 1. - 1.5 * Usq ) - 0.5*Phi000;
	rho_par *=0.25;
	f_eq100  = rho_par *( 1. + 3. * Ux*(1+1.5*(+Ux)) - 1.5*Usq) - 0.5*Phi100;		f_eq010  = rho_par *( 1. + 3. * Uy *(1+1.5*(+Uy)) - 1.5*Usq) - 0.5*Phi010;
	f_eq200  = rho_par *( 1.  - 3. * Ux*(1+1.5*(-Ux)) - 1.5*Usq) - 0.5*Phi200;		f_eq020  = rho_par *( 1.  - 3. * Uy*(1+1.5*(-Uy)) - 1.5*Usq) - 0.5*Phi020;
	f_eq001  = rho_par *( 1. + 3. * Uz*(1+1.5*(+Uz)) - 1.5*Usq) - 0.5*Phi001;		f_eq002  = rho_par *( 1.  - 3. * Uz*(1+1.5*(-Uz)) - 1.5*Usq) - 0.5*Phi002;
	rho_par *=0.25;
	f_eq110  = rho_par * ( 1. + 3.* ( Ux + Uy) * (1+1.5*(+Ux+Uy)) - 1.5*Usq)- 0.5*Phi110;	f_eq210 = rho_par * ( 1. + 3.* (-Ux + Uy) * (1+1.5*(-Ux	+Uy	)) - 1.5*Usq)- 0.5*Phi210;
	f_eq220  = rho_par * ( 1. + 3.* (-Ux - Uy)*(1+1.5*(-Ux-Uy)) - 1.5*Usq)- 0.5*Phi220;	f_eq120 = rho_par * ( 1. + 3.* ( Ux  - Uy) *(1+1.5*(	+Ux	-Uy	)) - 1.5*Usq)- 0.5*Phi120;
	f_eq101 = rho_par * ( 1. + 3.* ( Ux + Uz) *(1+1.5*(+Ux+Uz)) - 1.5*Usq)- 0.5*Phi101;	f_eq011 = rho_par * ( 1. + 3.* ( Uy + Uz) *(1+1.5*(	+Uy	+Uz	)) - 1.5*Usq)- 0.5*Phi011;
	f_eq201 = rho_par * ( 1. + 3.* (-Ux + Uz) *(1+1.5*(-Ux+Uz)) - 1.5*Usq)- 0.5*Phi201;	f_eq021 = rho_par * ( 1. + 3.* (-Uy + Uz) *(1+1.5*(	-Uy	+Uz	)) - 1.5*Usq)- 0.5*Phi021;
	f_eq102 = rho_par * ( 1. + 3.* ( Ux - Uz)*(1+1.5*(+Ux-Uz)) - 1.5*Usq)- 0.5*Phi102;	f_eq012 = rho_par * ( 1. + 3.* ( Uy  - Uz) *(1+1.5*(	+Uy	-Uz	)) - 1.5*Usq)- 0.5*Phi012;
	f_eq202 = rho_par * ( 1. + 3.* (-Ux - Uz)*(1+1.5*(-Ux-Uz)) - 1.5*Usq)- 0.5*Phi202;		f_eq022 = rho_par * ( 1. + 3.* (-Uy  - Uz) *(1+1.5*(	-Uy	-Uz	)) - 1.5*Usq)- 0.5*Phi022;
	rho_par *=0.25;
	f_eq111 = rho_par * ( 1. + 3.* (+Ux	 +Uy +Uz) *(1+1.5*(+Ux	+Uy	+Uz	)) - 1.5*Usq)- 0.5*Phi111;
	f_eq211 = rho_par * ( 1. + 3.* ( -Ux	 +Uy +Uz) *(1+1.5*(-Ux	+Uy	+Uz	)) - 1.5*Usq)- 0.5*Phi211;
	f_eq221 = rho_par * ( 1. + 3.* ( -Ux  -Uy 	+Uz) *(1+1.5*(-Ux	-Uy	+Uz	)) - 1.5*Usq)- 0.5*Phi221;
	f_eq121 = rho_par * ( 1. + 3.* (+Ux  -Uy  +Uz) *(1+1.5*(+Ux	-Uy	+Uz	)) - 1.5*Usq)- 0.5*Phi121;
	f_eq112 = rho_par * ( 1. + 3.* (+Ux  +Uy -Uz) *(1+1.5*(	+Ux	+Uy	-Uz	)) - 1.5*Usq)- 0.5*Phi112;
	f_eq212 = rho_par * ( 1. + 3.* ( -Ux	 +Uy -Uz) *(1+1.5*(	-Ux	+Uy	-Uz	)) - 1.5*Usq)- 0.5*Phi212;
	f_eq222 = rho_par * ( 1. + 3.* ( -Ux  -Uy	-Uz) *(1+1.5*(	-Ux	-Uy	-Uz	)) - 1.5*Usq)- 0.5*Phi222;
	f_eq122 = rho_par * ( 1. + 3.* (+Ux  -Uy	-Uz) *(1+1.5*(	+Ux	-Uy	-Uz	)) - 1.5*Usq)- 0.5*Phi122;
	
	/* set S tensor  from non-equilibrium populations in 2 steps: first, add standard popsulations, then subtract equilibrium ones */
	
	real_t Sxx	=	+f100+f200+f110	+f210+f220+f120+f101+f201		+f102		+f202		+f111	+f211	+f221	+f121	+f112	+f212	+f222	+f122;
	real_t Sxy	=	+f110-f210+f220-f120+f111-f211+f221-f121+f112	-f212	+f222	-f122;
	real_t Sxz	=	+f101-f201-f102+f202+f111-f211-f221+f121-f112	+f212	+f222	-f122;
	real_t Syy	=	+f010+f020+f110+f210	+f220+f120+f011	+f021+f012+f022+f111	+f211+f221	+f121	+f112	+f212	+f222	+f122;
	real_t Syz	=	+f011-f021-f012+f022+f111+f211-f221-f121-f112-f212+f222+f122;
	real_t Szz	=	+f001+f002+f101+f011	+f201+f021+f102+f012	+f202	+f022	+f111	+f211	+f221	+f121	+f112	+f212	+f222	+f122;

	 Sxx	-=	+f_eq100+f_eq200+f_eq110	+f_eq210+f_eq220+f_eq120+f_eq101+f_eq201		+f_eq102		+f_eq202		+f_eq111	+f_eq211	+f_eq221	+f_eq121	+f_eq112	+f_eq212	+f_eq222	+f_eq122;
	 Sxy	-=	+f_eq110-f_eq210+f_eq220-f_eq120+f_eq111-f_eq211+f_eq221-f_eq121+f_eq112	-f_eq212	+f_eq222	-f_eq122;
	 Sxz	-=	+f_eq101-f_eq201-f_eq102+f_eq202+f_eq111-f_eq211-f_eq221+f_eq121-f_eq112	+f_eq212	+f_eq222	-f_eq122;
	 Syy	-=	+f_eq010+f_eq020+f_eq110+f_eq210	+f_eq220+f_eq120+f_eq011	+f_eq021+f_eq012+f_eq022+f_eq111	+f_eq211+f_eq221	+f_eq121	+f_eq112	+f_eq212	+f_eq222	+f_eq122;
	 Syz	-=	+f_eq011-f_eq021-f_eq012+f_eq022+f_eq111+f_eq211-f_eq221-f_eq121-f_eq112-f_eq212+f_eq222+f_eq122;
	 Szz	-=	+f_eq001+f_eq002+f_eq101+f_eq011	+f_eq201+f_eq021+f_eq102+f_eq012	+f_eq202	+f_eq022	+f_eq111	+f_eq211	+f_eq221	+f_eq121	+f_eq112	+f_eq212	+f_eq222	+f_eq122;
	 
	real_t Syx	=	Sxy ;
	real_t Szx	=	Sxz;
	real_t Szy	=	Syz;
	 
	  /* Only deviatoric part of S tensor is used */
	 real_t traceb3 = (Sxx+ Syy+Szz)/3.; //trace divided by 3
	 Sxx-=traceb3;
	 Syy-=traceb3;
	 Szz-=traceb3;

	/* contraction of S */
	real_t Scontr = Sxx*Sxx + Sxy*Sxy + Sxz*Sxz + Syx*Syx + Syy*Syy + Syz*Syz + Szx*Szx + Szy*Szy + Szz*Szz;

	if(Scontr < 2*YieldStress*YieldStress){
		yield_stat =  1.0;
		nu_app    = 0.0;
	}
	else{
		yield_stat = 0.0;
		real_t omega= 1.0/(3*nu+0.5);
		real_t sq2s = sqrt(2./Scontr);
		real_t c = (6*nu - 1.0)/(6*nu + 1.0)+ sq2s * YieldStress * omega;
		if( YieldStress < 1e-15 ) c = (6*nu - 1.0)/(6*nu + 1.0); // for security if the viscoplastic model is used with close-to-zero yield stress
		Sxx *=c; Sxy *=c; Sxz *=c; Syx *=c; Syy *=c; Syz *=c; Szx *=c; Szy *=c; Szz *=c;
		nu_app= nu + YieldStress/sq2s; //apparent viscosity
	}

	/* update populations with F_eq + Phi + F_sigma */

	f000 =f_eq000+Phi000 ;
	f200 = 1./3.*( Sxx)+f_eq200+Phi200 ;		 f100 	= 1./3. * (Sxx)+f_eq100+Phi100 ;
	f020 = 1./3.*(+Syy)+f_eq020+Phi020 ;		 f010 	= 1./3.*(+Syy)+f_eq010+Phi010 ;
	f002 = 1./3.*(+Szz)+f_eq002+Phi002 ;		 f001 	= 1./3.*(+Szz)+f_eq001+Phi001 ;
		
	f220= 1./12.*(Sxx+2*Sxy+Syy)+f_eq220+Phi220 ;		 f110 = 1./12.*(Sxx+2*Sxy+Syy)+f_eq110+Phi110 ;
	f120 = 1./12.*(Sxx-2*Sxy+Syy)+f_eq120+Phi120 ;		 f210 = 1./12.*( Sxx-2*Sxy+Syy)+f_eq210+Phi210 ;
	f202  = 1./12.*(Sxx+2*Sxz+Szz)+f_eq202+Phi202 ;	 f101 = 1./12.*(Sxx+2*Sxz+Szz)+f_eq101+Phi101 ;
	f022	  = 1./12.*(+Syy+2*Syz+Szz)+f_eq022+Phi022 ;	 f011 = 1./12.*(+Syy+2*Syz+Szz)+f_eq011+Phi011 ;
	f102  =  1./12.*(Sxx-2*Sxz+Szz)+f_eq102+Phi102 ; 	 f201 = 1./12.*(Sxx-2*Sxz+Szz)+f_eq201+Phi201 ;
	f012 = 1./12.*(+Syy-2*Syz+Szz)+f_eq012+Phi012 ;     	 f021 = 1./12.*(+Syy-2*Syz+Szz)+f_eq021+Phi021 ;
	
	f222    = 1./48.*(Sxx+2*Sxy+2*Sxz+Syy+2*Syz+Szz)+f_eq222+Phi222; f111 = 1./48.*(Sxx+2*Sxy+2*Sxz+Syy+2*Syz+Szz)+f_eq111 +Phi111  ;
	f122  = 1./48.*(Sxx-2*Sxy-2*Sxz+Syy+2*Syz+Szz)+f_eq122+Phi122 ;	 f211 = 1./48.*(Sxx-2*Sxy-2*Sxz+Syy+2*Syz+Szz)+f_eq211+Phi211 ;
	f112 = 1./48.*(Sxx+2*Sxy-2*Sxz+Syy-2*Syz+Szz)+f_eq112+Phi112;	 f221 = 1./48.*(Sxx+2*Sxy-2*Sxz+Syy-2*Syz+Szz)+f_eq221+Phi221 ;
	f212  = 1./48.*(Sxx-2*Sxy+2*Sxz+Syy-2*Syz+Szz)+f_eq212+Phi212 ;	 f121 = 1./48.*(Sxx-2*Sxy+2*Sxz+Syy-2*Syz+Szz)+f_eq121+Phi121 ;
	
	switch (NodeType & NODE_ADDITIONALS) {
		
		case NODE_XYslice1:
		AddToXYvx(Ux);
		AddToXYvy(Uy);
		AddToXYvz(Uz);
		AddToXYrho1(Rho);
		AddToXYarea(1);
		break;
		
		case NODE_XYslice2:
		AddToXYrho2(Rho);
		break;
		
		case NODE_XZslice1:
		AddToXZvx(Ux);
		AddToXZvy(Uy);
		AddToXZvz(Uz);
		AddToXZrho1(Rho);
		AddToXZarea(1);
		break;
		
		case NODE_XZslice2:
		AddToXZrho2(Rho);
		break;
		
		case NODE_YZslice1:
		AddToYZvx(Ux);
		AddToYZvy(Uy);
		AddToYZvz(Uz);
		AddToYZrho1(Rho);
		AddToYZarea(1);
		break;
		
		case NODE_YZslice2:
		AddToYZrho2(Rho);
		break;
		
	}
	
}

