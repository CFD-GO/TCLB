#define CudaDeviceFunction 
#define real_t double
#include <math.h>
#include <assert.h>
#include <stdio.h>
//x4 x5 x6 x7 
// 4.269327243786159798e+00 -2.705207483582461315e-13 4.313636855125160707e+00 1.000000000000000000e+01 8
// p -2.047304e+05 4.115177e+07 -4.115166e+07 -1.028791e+02 
// m 4.269548e+00 -1.338658e-09 4.313626e+00 4.313648e-05 

// const real_t w0 = 0.1;//4.269327243786159798e+00;
// const real_t s0 = 10;// -2.705207483582461315e-13; 
// const real_t i0 = 0.1;//4.313636855125160707e+00; 
// const real_t r0 = 1e-15;///3.235241e-05; 
// const real_t n0 = s0+r0+i0;//4.313669207536073635e+00;

// const real_t C_1 = 1;//4.213999999999999972e-05; 
// const real_t C_2 = 1;//1.000000000000000021e-02; 
// const real_t C_3 = 1;//5.000000000000000409e-06;


const real_t w0 = 4.269327243786159798e+00;
const real_t s0 = -2.705207483582461315e-13; 
const real_t i0 = 4.313636855125160707e+00; 
const real_t r0 = 3.235241e-05; 
const real_t n0 = 4.313669207536073635e+00;

const real_t C_1 = 4.213999999999999972e-05; 
const real_t C_2 = 1.000000000000000021e-02; 
const real_t C_3 = 5.000000000000000409e-06;

CudaDeviceFunction void localCalcQ(real_t* localPhi, real_t* localQ, real_t N) 
{	
    
    const real_t Beta = C_1;
    const real_t Beta_w = C_2; 
    const real_t Gamma = C_3;



    const real_t W = localPhi[0];
    const real_t S = localPhi[1];
    const real_t I = localPhi[2];
    //const real_t R = localPhi[3];

    //const real_t N = localPhi[4];

    localQ[0] =   Beta_w*(I-W);
    localQ[1] =  -Beta*S*W / N;
    localQ[2] =   (Beta*S*W/N - Gamma*I); 
    //localQ[3] =   Gamma*I;
    //localQ[4] = 0;		 //N
    //printf("Phi %f %f %f %f\n", localPhi[0], localPhi[1], localPhi[2], localPhi[3]);
    //printf("Q %f %f %f %f\n", q[0], q[1], q[2], q[3]);
}

#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;
 
CudaDeviceFunction int main()
{

	const real_t x0 = w0;
	const real_t x1 = s0;
	const real_t x2 = i0;
    real_t X1[3], X2[3], f1[3], f2[3];

    // X1[0] = w0;
    // X1[1] = s0;
    // X1[2] = i0;
    
    // localCalcQ(X1, f1, n0);
    // X1[0] = X1[0] - f1[0]/2.;
    // X1[1] = X1[1] - f1[1]/2.;
    // X1[2] = X1[2] - f1[2]/2.;
    
    // const real_t x0 = X1[0];
    // const real_t x1 = X1[1];
    // const real_t x2 = X1[2];


    const real_t Beta = C_1;
    const real_t Beta_w = C_2; 
    const real_t gamma = C_3;
    const real_t N = n0;

	Matrix3d Jacobian;

	Vector3d X, dX, X0, F;
	X(0) = x0/N;
	X(1) = x1/N;
	X(2) = x2/N;

	X0 = X;
	cout << "Newton Init " << X.transpose() << endl;

	for (int i=0; i < 100; i++) {
		Jacobian(0,0) =  -1.0/2.0*Beta_w - 1 ;
		Jacobian(0,1) =  -1.0/2.0*Beta*X(1) ;
		Jacobian(0,2) =  (1.0/2.0)*Beta*X(1) ;
		// Jacobian(0,3) =  0 ;
		Jacobian(1,0) =  0 ;
		Jacobian(1,1) =  -1.0/2.0*Beta*X(0) - 1 ;
		Jacobian(1,2) =  (1.0/2.0)*Beta*X(0) ;
		// Jacobian(1,3) =  0 ;
		Jacobian(2,0) =  (1.0/2.0)*Beta_w ;
		Jacobian(2,1) =  0 ;
		Jacobian(2,2) =  -1.0/2.0*gamma - 1 ;
		// Jacobian(2,3) =  (1.0/2.0)*gamma ;
		// Jacobian(3,0) =  0 ;
		// Jacobian(3,1) =  0 ;
		// Jacobian(3,2) =  0 ;
		// Jacobian(3,3) =  -1 ;

		F(0) =  (1.0/2.0)*Beta_w*(-X(0) + X(2)) - X(0) + X0(0) ;
		F(1) =  -1.0/2.0*Beta*X(0)*X(1) - X(1) + X0(1) ;
		F(2) =  (1.0/2.0)*Beta*X(0)*X(1) - 1.0/2.0*X(2)*gamma - X(2) + X0(2) ;
		// F(3) =  (1.0/2.0)*X(2)*gamma - X(3) + X0(3) ;

		dX = Jacobian.householderQr().solve(-F);
		// cout << "Newton Error " << F.norm() << endl;
		X = X + dX;
		if (F.norm() < 1E-16) {
			cout << "Converged  in "<< i << endl;
			break;
		}
		cout << "Newton Step " << X.transpose() << endl;

		// cout << "Newton dX " << dX.transpose() << endl;
		
		// cout << "Newton Step " << X.transpose() << endl;
	}
	cout << "Newton Error " << F.norm() << endl;
	cout << "Newton Result " << N*X.transpose() << endl<< endl<< endl;
	// x4 x5 x6 x7 4.313669e+00 4.214000e-05 1.000000e-02 5.000000e-06
	// s 4.269327e+00 -2.705207e-13 4.313637e+00 3.235241e-05


    // Try Secant

    





    // END FAKE INTI

    X1[0] = fabs(x0);
    X1[1] = fabs(x1);
    X1[2] = fabs(x2);

    localCalcQ(X1, f1, n0);

    X2[0] = fabs(X1[0] + f1[0]/2.);
    X2[1] = fabs(X1[1] + f1[1]/2.);
    X2[2] = fabs(X1[2] + f1[2]/2.);

    printf("Secant 0: X1 %e %e %e  \n", X1[0], X1[1], X1[2]);
    printf("Secant 0: X2 %e %e %e  \n", X2[0], X2[1], X2[2]);

    for(int i =0; i < 20; i++) {
        localCalcQ(X1, f1, n0);
        // printf("Newton  Q1 %e %e %e %e \n", f1[0], f1[1], f1[2]);
 
        f1[0] = x0 + f1[0]/2. - X1[0]; 
        f1[1] = x1 + f1[1]/2. - X1[1];
        f1[2] = x2 + f1[2]/2. - X1[2];


        real_t epsilon_step = 0;
        for(int k=0; k < 3; k++){
            if ( fabs(f1[k]) > epsilon_step ) {
                epsilon_step = fabs(f1[k]);
            }
        }
        if (epsilon_step < 1E-12){

            for(int k=0; k < 3; k++){
                X1[k] = fabs(X1[k]);
            }
            printf("Secant result: %e %e %e EPS: %e\n", X1[0], X1[1], X1[2], epsilon_step);                    
            printf("CONVERGED\n\n");
            break;
        }


        // printf("Newton  F1 %e %e %e \n", f1[0], f1[1], f1[2]);

        localCalcQ(X2, f2, n0);
        // printf("Newton  Q2 %e %e %e \n", f2[0], f2[1], f2[2]);
        f2[0] = x0 + f2[0]/2. - X2[0]; 
        f2[1] = x1 + f2[1]/2. - X2[1];
        f2[2] = x2 + f2[2]/2. - X2[2];
        
        // printf("Newton  F2 %e %e %e \n", f2[0], f2[1], f2[2]);

        //printf("Iter %e %e %e %e \n", f1[0], f1[1], f1[2], f1[3]);
       // printf("Newton  %e %e %e %e \n", X2[0]-x0, X2[1]-x1, X2[2]-x2, X2[3]-x3);

        for(int k=0; k < 3; k++){
            const real_t tt = X1[k];
            X1[k] = X1[k] - f1[k] * (X1[k] - X2[k]) / (f1[k] - f2[k]);
            X2[k] = tt;
        }

        printf("Secant  X1 %e %e %e EPS: %e\n", X1[0], X1[1], X1[2], epsilon_step);

        printf("STEP\n\n");


    }


    //printf("x4 x5 x6 x7 %e %e %e %e\n", x4, x5, x6, x7);
    //printf("m %e %e %e %e \n", X2[0]-x0, X2[1]-x1, X2[2]-x2, X2[3]-x3);

        
    
    return 0;

}


// CudaDeviceFunction void CalcQ() 
// {	
    
//     const real_t Beta = C_1;
//     const real_t Beta_w = C_2; 
//     const real_t Gamma = C_3;



//     const real_t W = phi[0];
//     const real_t S = phi[1];
//     const real_t I = phi[2];
//     const real_t R = phi[3];

//     const real_t N = phi[4];


//     q[0] =   Beta_w*(I-W);
//     q[1] =  -Beta*S*W / N;
//     q[2] =   (Beta*S*W/N - Gamma*I); 
//     q[3] =   Gamma*I;
//     q[4] = 0;		 //N
//     //printf("Phi %f %f %f %f\n", phi[0], phi[1], phi[2], phi[3]);
//     //printf("Q %f %f %f %f\n", q[0], q[1], q[2], q[3]);
// }


// x4 x5 x6 x7 4.313669e+00 4.214000e-05 1.000000e-02 5.000000e-06
// s 4.269327e+00 -2.705207e-13 4.313637e+00 1.000000e+01 
// p -2.047304e+05 4.115177e+07 -4.115166e+07 -1.028791e+02 
// m 4.269548e+00 -1.338658e-09 4.313626e+00 4.313648e-05 


// x4 x5 x6 x7 4.313669e+00 4.214000e-05 1.000000e-02 5.000000e-06
// s 4.269327e+00 -2.705207e-13 4.313637e+00 1.000000e+01 
// p -2.047304e+05 4.115177e+07 -4.115166e+07 -1.028791e+02 
// m 4.269547e+00 2.459523e-09 4.313626e+00 4.313648e-05 


//    // phi[4] = x4; // N
//     // x0 = W^\star
//     // x1 = S^\star
//     // x2 = I^\star
//     // x3 = R^\star

//     // x4 = N
//     //const real_t x4 = x1 + x2 + x3;

//     // x5 = Beta
//     const real_t x5 = C_1;

//     // x6 = Beta_w
//     const real_t x6 = C_2;

//     // x7 = gamma
//     const real_t x7 = C_3;

    
//     const real_t  x8 = x0*x5 ; // 1
//     const real_t  x9 = 2*x8 ; // 1
//     const real_t  x10 = x7*x8 ; // 1
//     const real_t  x11 = pow(x4, 2) ; // 1
//     const real_t  x12 = 16*x11 ; // 1
//     const real_t  x13 = x4*x8 ; // 1
//     const real_t  x14 = x5*x6 ; // 1
//     const real_t  x15 = x1*x14 ; // 1
//     const real_t  x16 = 8*x4 ; // 1
//     const real_t  x17 = x14*x2 ; // 1
//     const real_t  x18 = x6*x7 ; // 1
//     const real_t  x19 = x18*x4 ; // 1
//     const real_t  x20 = 4*x4 ; // 1
//     const real_t  x21 = x18*x5 ; // 1
//     const real_t  x22 = x1*x21 ; // 1
//     const real_t  x23 = x2*x21 ; // 1
//     const real_t  x24 = pow(x5, 2) ; // 1
//     const real_t  x25 = pow(x0, 2)*x24 ; // 2
//     const real_t  x26 = 4*x25 ; // 1
//     const real_t  x27 = pow(x6, 2) ; // 1
//     const real_t  x28 = 4*x11 ; // 1
//     const real_t  x29 = x27*x28 ; // 1
//     const real_t  x30 = pow(x7, 2) ; // 1
//     const real_t  x31 = x28*x30 ; // 1
//     const real_t  x32 = x0*x24 ; // 1
//     const real_t  x33 = x1*x32 ; // 1
//     const real_t  x34 = 4*x6 ; // 1
//     const real_t  x35 = x2*x32 ; // 1
//     const real_t  x36 = x30*x8 ; // 1
//     const real_t  x37 = x27*x5 ; // 1
//     const real_t  x38 = x20*x37 ; // 1
//     const real_t  x39 = 2*x6 ; // 1
//     const real_t  x40 = x39*x7 ; // 1
//     const real_t  x41 = x39*x4 ; // 1
//     const real_t  x42 = 2*x7 ; // 1
//     const real_t  x43 = x4*x42 ; // 1
//     const real_t  x44 = x37*x43 ; // 1
//     const real_t  x45 = x24*x27 ; // 1
//     const real_t  x46 = sqrt(pow(x1, 2)*x45 + 2*x1*x2*x45 - x1*x38 - x1*x44 + 16*x10*x4 + x11*x27*x30 + x12*x18 + x12*x6 + x12*x7 + x12 + 8*x13*x6 + 16*x13 - x15*x16 + x16*x17 + 8*x19*x8 + pow(x2, 2)*x45 + x2*x38 + x2*x44 - x20*x22 + x20*x23 + x20*x36 + x25*x30 + x26*x7 + x26 + x29*x7 + x29 + x31*x6 + x31 + x33*x34 + x33*x40 + x34*x35 + x35*x40 + x36*x41) ; // 71
//     const real_t  x47 = x19 + x20 + x41 + x43 ; // 3
//     const real_t  x48 = x46 + x47 ; // 1
//     const real_t  x49 = -x15 - x17 ; // 2
//     const real_t  x50 = x48 + x49 ; // 1
//     const real_t  x51 = 1.0/x5 ; // 1
//     const real_t  x52 = x51/(x18 + x39 + x42 + 4) ; // 4
//     const real_t  x53 = x10 + x9 ; // 1
//     const real_t  x54 = x15 + x17 + x53 ; // 2
//     const real_t  x55 = x51/x6 ; // 1
//     const real_t  x56 = (1.0/2.0)*x55 ; // 1
//     const real_t  x57 = 1.0/(x7 + 2) ; // 2
//     const real_t  x58 = x55*x57 ; // 1
//     const real_t  x59 = x20*x7 ; // 1
//     const real_t  x60 = x42*x8 ; // 1
//     const real_t  x61 = 4*x14*x3 ; // 2
//     const real_t  x62 = x41*x7 ; // 1
//     const real_t  x63 = x30*x4 ; // 1
//     const real_t  x64 = 2*x63 ; // 1
//     const real_t  x65 = x6*x63 ; // 1
//     const real_t  x66 = x3*x40*x5 ; // 2
//     const real_t  x67 = x46*x7 ; // 1
//     const real_t  x68 = x56*x57 ; // 1
//     const real_t  x69 = -x46 + x47 ; // 1
//     const real_t  Wp = x52*(x10 - x50 + x9) ; // 3
//     const real_t  Sp = x56*(x48 + x54) ; // 2
//     const real_t  Ip = -x58*(x50 + x53) ; // 3
//     //const real_t  Rp = x68*(x22 + x23 - x36 - x59 - x60 + x61 - x62 - x64 - x65 + x66 - x67) ; // 11
//     const real_t  Wm = x52*(-x19 - x20 - x41 - x43 + x46 + x54) ; // 6
//     const real_t  Sm = x56*(x54 + x69) ; // 2
//     const real_t  Im = -x58*(x49 + x53 + x69) ; // 4
//     //const real_t  Rm = x68*(x22 + x23 - x36 - x59 - x60 + x61 - x62 - x64 - x65 + x66 + x67) ; // 11
//     // Opers =  255

//     const real_t  Rs = r0;
//     const real_t  Rp = Ip * x7 / 2. + Rs;
//     const real_t  Rm = Im * x7 / 2. + Rs;

//     const real_t eps = -1E-10;

//     printf("x4 x5 x6 x7 %e %e %e %e\n", x4, x5, x6, x7);
//     printf("s %e %e %e %e \n", x0, x1, x2, x3);
//     printf("p %e %e %e %e \n", Wp, Sp, Ip, Rp);
//     printf("m %e %e %e %e \n", Wm, Sm, Im, Rm);






/**
	CudaDeviceFunction void CalcPhi() 
	{	
		const real_t x0 = <?%s  C(sum(fs[[1]])) ?>;
		const real_t x1 = <?%s  C(odes[[1]]) ?>(0,0);
		const real_t x2 = <?%s  C(odes[[2]]) ?>(0,0);
		const real_t x3 = 10.;
		const real_t x4 = <?%s  C(odes[[4]]) ?>(0,0);

        phi[4] = x4; // N
		// x0 = W^\star
		// x1 = S^\star
		// x2 = I^\star
		// x3 = R^\star

		// x4 = N
		//const real_t x4 = x1 + x2 + x3;

		// x5 = Beta
		const real_t x5 = C_1;

		// x6 = Beta_w
		const real_t x6 = C_2;

		// x7 = gamma
		const real_t x7 = C_3;

		
		const real_t  x8 = x0*x5 ; // 1
		const real_t  x9 = 2*x8 ; // 1
		const real_t  x10 = x7*x8 ; // 1
		const real_t  x11 = pow(x4, 2) ; // 1
		const real_t  x12 = 16*x11 ; // 1
		const real_t  x13 = x4*x8 ; // 1
		const real_t  x14 = x5*x6 ; // 1
		const real_t  x15 = x1*x14 ; // 1
		const real_t  x16 = 8*x4 ; // 1
		const real_t  x17 = x14*x2 ; // 1
		const real_t  x18 = x6*x7 ; // 1
		const real_t  x19 = x18*x4 ; // 1
		const real_t  x20 = 4*x4 ; // 1
		const real_t  x21 = x18*x5 ; // 1
		const real_t  x22 = x1*x21 ; // 1
		const real_t  x23 = x2*x21 ; // 1
		const real_t  x24 = pow(x5, 2) ; // 1
		const real_t  x25 = pow(x0, 2)*x24 ; // 2
		const real_t  x26 = 4*x25 ; // 1
		const real_t  x27 = pow(x6, 2) ; // 1
		const real_t  x28 = 4*x11 ; // 1
		const real_t  x29 = x27*x28 ; // 1
		const real_t  x30 = pow(x7, 2) ; // 1
		const real_t  x31 = x28*x30 ; // 1
		const real_t  x32 = x0*x24 ; // 1
		const real_t  x33 = x1*x32 ; // 1
		const real_t  x34 = 4*x6 ; // 1
		const real_t  x35 = x2*x32 ; // 1
		const real_t  x36 = x30*x8 ; // 1
		const real_t  x37 = x27*x5 ; // 1
		const real_t  x38 = x20*x37 ; // 1
		const real_t  x39 = 2*x6 ; // 1
		const real_t  x40 = x39*x7 ; // 1
		const real_t  x41 = x39*x4 ; // 1
		const real_t  x42 = 2*x7 ; // 1
		const real_t  x43 = x4*x42 ; // 1
		const real_t  x44 = x37*x43 ; // 1
		const real_t  x45 = x24*x27 ; // 1
		const real_t  x46 = sqrt(pow(x1, 2)*x45 + 2*x1*x2*x45 - x1*x38 - x1*x44 + 16*x10*x4 + x11*x27*x30 + x12*x18 + x12*x6 + x12*x7 + x12 + 8*x13*x6 + 16*x13 - x15*x16 + x16*x17 + 8*x19*x8 + pow(x2, 2)*x45 + x2*x38 + x2*x44 - x20*x22 + x20*x23 + x20*x36 + x25*x30 + x26*x7 + x26 + x29*x7 + x29 + x31*x6 + x31 + x33*x34 + x33*x40 + x34*x35 + x35*x40 + x36*x41) ; // 71
		const real_t  x47 = x19 + x20 + x41 + x43 ; // 3
		const real_t  x48 = x46 + x47 ; // 1
		const real_t  x49 = -x15 - x17 ; // 2
		const real_t  x50 = x48 + x49 ; // 1
		const real_t  x51 = 1.0/x5 ; // 1
		const real_t  x52 = x51/(x18 + x39 + x42 + 4) ; // 4
		const real_t  x53 = x10 + x9 ; // 1
		const real_t  x54 = x15 + x17 + x53 ; // 2
		const real_t  x55 = x51/x6 ; // 1
		const real_t  x56 = (1.0/2.0)*x55 ; // 1
		const real_t  x57 = 1.0/(x7 + 2) ; // 2
		const real_t  x58 = x55*x57 ; // 1
		const real_t  x59 = x20*x7 ; // 1
		const real_t  x60 = x42*x8 ; // 1
		const real_t  x61 = 4*x14*x3 ; // 2
		const real_t  x62 = x41*x7 ; // 1
		const real_t  x63 = x30*x4 ; // 1
		const real_t  x64 = 2*x63 ; // 1
		const real_t  x65 = x6*x63 ; // 1
		const real_t  x66 = x3*x40*x5 ; // 2
		const real_t  x67 = x46*x7 ; // 1
		const real_t  x68 = x56*x57 ; // 1
		const real_t  x69 = -x46 + x47 ; // 1
		const real_t  Wp = x52*(x10 - x50 + x9) ; // 3
		const real_t  Sp = x56*(x48 + x54) ; // 2
		const real_t  Ip = -x58*(x50 + x53) ; // 3
		//const real_t  Rp = x68*(x22 + x23 - x36 - x59 - x60 + x61 - x62 - x64 - x65 + x66 - x67) ; // 11
		const real_t  Wm = x52*(-x19 - x20 - x41 - x43 + x46 + x54) ; // 6
		const real_t  Sm = x56*(x54 + x69) ; // 2
		const real_t  Im = -x58*(x49 + x53 + x69) ; // 4
		//const real_t  Rm = x68*(x22 + x23 - x36 - x59 - x60 + x61 - x62 - x64 - x65 + x66 + x67) ; // 11
		// Opers =  255

		const real_t  Rs = <?%s  C(odes[[3]]) ?>(0,0);
		const real_t  Rp = Ip * x7 / 2. + Rs;
		const real_t  Rm = Im * x7 / 2. + Rs;

		const real_t eps = -1E-10;
		if ( Sp >=eps && Ip >=eps && Rp >=eps && Wp >=eps) {
			phi[1] = Sp < 0 ? 0 : Sp;
			phi[2] = Ip < 0 ? 0 : Ip;
			phi[3] = Rp < 0 ? 0 : Rp;
			phi[0] = Wp < 0 ? 0 : Wp;

		} else if ( Sm >=eps&& Im >=eps && Rm >=eps && Wm >=eps ) {
			phi[1] = Sm < 0 ? 0 : Sm;
			phi[2] = Im < 0 ? 0 : Im;
			phi[3] = Rm < 0 ? 0 : Rm;
			phi[0] = Wm < 0 ? 0 : Wm;

		} else {
			printf("x4 x5 x6 x7 %.18e %.18e %.18e %.18e\n", x4, x5, x6, x7);
			printf("s %.18e %.18e %.18e %.18e %ld\n", x0, x1, x2, x3,sizeof(real_t));
			printf("p %e %e %e %e \n", Wp, Sp, Ip, Rp);
			printf("m %e %e %e %e \n", Wm, Sm, Im, Rm);

			// Try Secant

			real_t X1[5], X2[5], f1[5], f2[5];
			
			X1[0] = x0;
			X1[1] = x1;
			X1[2] = x2;
			X1[3] = 1;
			X1[4] = phi[4];

			X2[0] = x0+0.001;
			X2[1] = x1;
			X2[2] = x2;
			X2[3] = 1;
			X2[4] = phi[4];

			printf("Newton 0 %e %e %e %e \n", X2[0], X2[1], X2[2], X2[3]);

			for(int i =0; i < 50; i++) {
				localCalcQ(X1, f1);
				localCalcQ(X2, f2);
				f1[0] = x0 - f1[0]/2. - X1[0]; 
				f1[1] = x1 - f1[1]/2. - X1[1];
				f1[2] = x2 - f1[2]/2. - X1[2];

				f2[0] = x0 - f2[0]/2. - X2[0]; 
				f2[1] = x1 - f2[1]/2. - X2[1];
				f2[2] = x2 - f2[2]/2. - X2[2];

				printf("Iter %e %e %e %e \n", f1[0], f1[1], f1[2], f1[3]);

				for(int k=0; k < 3; k++){
					const real_t tt = X1[k];
					X1[k] = X2[k] - f1[k] * (X2[k] - X1[k]) / (f2[k] - f1[k]);
					X2[k] = tt;
				}




				printf("m %e %e %e %e \n", X2[0], X2[1], X2[2], X2[3]);

			}


			printf("x4 x5 x6 x7 %e %e %e %e\n", x4, x5, x6, x7);
			printf("m %e %e %e %e \n", X2[0], X2[1], X2[2], X2[3]);

			assert(0);
		}


	}

	CudaDeviceFunction void localCalcQ(real_t* localPhi, real_t* localQ) 
	{	
		
		const real_t Beta = C_1;
		const real_t Beta_w = C_2; 
		const real_t Gamma = C_3;

 

		const real_t W = localPhi[0];
		const real_t S = localPhi[1];
		const real_t I = localPhi[2];
		const real_t R = localPhi[3];

		const real_t N = localPhi[4];

		localQ[0] =   Beta_w*(I-W);
		localQ[1] =  -Beta*S*W / N;
		localQ[2] =   (Beta*S*W/N - Gamma*I); 
		localQ[3] =   Gamma*I;
		localQ[4] = 0;		 //N
		//printf("Phi %f %f %f %f\n", localPhi[0], localPhi[1], localPhi[2], localPhi[3]);
		//printf("Q %f %f %f %f\n", q[0], q[1], q[2], q[3]);
	}
**/



/**				
		// poor man`s secant method		
		real_t X1[3], X2[3], f1[3], f2[3];
		

		X1[0] = fabs(x0);
		X1[1] = fabs(x1);
		X1[2] = fabs(x2);

		localCalcQ(X1, f1, x4);

		X2[0] = fabs(X1[0] + f1[0]/2.);
		X2[1] = fabs(X1[1] + f1[1]/2.);
		X2[2] = fabs(X1[2] + f1[2]/2.);

		for(int i =0; i < 50; i++) {
			localCalcQ(X1, f1, x4);
	
			f1[0] = x0 + f1[0]/2. - X1[0]; 
			f1[1] = x1 + f1[1]/2. - X1[1];
			f1[2] = x2 + f1[2]/2. - X1[2];

			real_t epsilon_step = 0;
			for(int k=0; k < 3; k++){
				if ( fabs(f1[k]) > epsilon_step ) {
					epsilon_step = fabs(f1[k]);
				}
			}
			if (epsilon_step < 1E-8){

				for(int k=0; k < 3; k++){
					X1[k] = fabs(X1[k]);
				}
				phi[0] = X1[0];
				phi[1] = X1[1];
				phi[2] = X1[2];
				phi[3] = X1[2]*C_3/2. + x3;
							
				return;
			}

			localCalcQ(X2, f2, x4);

			f2[0] = x0 + f2[0]/2. - X2[0]; 
			f2[1] = x1 + f2[1]/2. - X2[1];
			f2[2] = x2 + f2[2]/2. - X2[2];
			
			for(int k=0; k < 3; k++){
				const real_t tt = X1[k];
				X1[k] = X1[k] - f1[k] * (X1[k] - X2[k]) / (f1[k] - f2[k]);
				X2[k] = tt;
			}

		}
*/
		