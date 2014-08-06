
#define S2 1.3333
#define S3 1.0
#define S5 1.0
#define S7 1.0
#define S8 omega
#define S9 omega

CudaDeviceFunction real_t getRho()
{
    return f[8] + f[7] + f[6] + f[5] + f[4] + f[3] + f[2] + f[1] + f[0];
}

CudaDeviceFunction real_t getT()
{
    return T[8] + T[7] + T[6] + T[5] + T[4] + T[3] + T[2] + T[1] + T[0];
}

CudaDeviceFunction real_t getQ()
{
    real_t u[2], usq, d, R[6], uf;
    d = f[8] + f[7] + f[6] + f[5] + f[4] + f[3] + f[2] + f[1] + f[0];
    u[0] = f[8] - f[7] - f[6] + f[5] - f[3] + f[1];
    u[1] = -f[8] - f[7] + f[6] + f[5] - f[4] + f[2];
    R[0] = -f[4] - f[3] - f[2] - f[1] + (f[8] + f[7] + f[6] + f[5] - f[0] * 2.) * 2.;
    /*R[1] = f[8] + f[7] + f[6] + f[5] + (-f[4] - f[3] - f[2] - f[1] + f[0] * 2.) * 2.;
    R[2] = f[8] - f[7] - f[6] + f[5] + (f[3] - f[1]) * 2.;
    R[3] = -f[8] - f[7] + f[6] + f[5] + (f[4] - f[2]) * 2.;*/
    R[4] = -f[4] + f[3] - f[2] + f[1];
    R[5] = -f[8] + f[7] - f[6] + f[5];
    usq = u[1] * u[1] + u[0] * u[0];

    R[0] = R[0] - (-2. * d + 3. * usq);
    /*R[1] = R[1] - (d - 3. * usq);
    R[2] = R[2] - (-u[0]);
    R[3] = R[3] - (-u[1]);*/
    R[4] = R[4] - (u[0] * u[0] - u[1] * u[1]);
    R[5] = R[5] - (u[0] * u[1]);
	real_t Q=0;
	Q += 2.*R[5]*R[5];
	Q += (R[0]*R[0] + 9.*R[4]*R[4])/18.;
	return Q;
}

CudaDeviceFunction real_t getQxx()
{
    real_t u[2], usq, d, R[6], uf;
    d = f[8] + f[7] + f[6] + f[5] + f[4] + f[3] + f[2] + f[1] + f[0];
    u[0] = f[8] - f[7] - f[6] + f[5] - f[3] + f[1];
    u[1] = -f[8] - f[7] + f[6] + f[5] - f[4] + f[2];
    R[0] = -f[4] - f[3] - f[2] - f[1] + (f[8] + f[7] + f[6] + f[5] - f[0] * 2.) * 2.;
    /*R[1] = f[8] + f[7] + f[6] + f[5] + (-f[4] - f[3] - f[2] - f[1] + f[0] * 2.) * 2.;
    R[2] = f[8] - f[7] - f[6] + f[5] + (f[3] - f[1]) * 2.;
    R[3] = -f[8] - f[7] + f[6] + f[5] + (f[4] - f[2]) * 2.;*/
    R[4] = -f[4] + f[3] - f[2] + f[1];
    R[5] = -f[8] + f[7] - f[6] + f[5];
    usq = u[1] * u[1] + u[0] * u[0];

    R[0] = R[0] - (-2. * d + 3. * usq);
    /*R[1] = R[1] - (d - 3. * usq);
    R[2] = R[2] - (-u[0]);
    R[3] = R[3] - (-u[1]);*/
    R[4] = R[4] - (u[0] * u[0] - u[1] * u[1]);
    R[5] = R[5] - (u[0] * u[1]);
	real_t Qxx=0;
	Qxx += (-0.02*(3*omega)/2)*(R[0]/6+R[4]/2);
	return Qxx;
}

CudaDeviceFunction real_t getQxy()
{
    real_t u[2], usq, d, R[6], uf;
    d = f[8] + f[7] + f[6] + f[5] + f[4] + f[3] + f[2] + f[1] + f[0];
    u[0] = f[8] - f[7] - f[6] + f[5] - f[3] + f[1];
    u[1] = -f[8] - f[7] + f[6] + f[5] - f[4] + f[2];
    R[0] = -f[4] - f[3] - f[2] - f[1] + (f[8] + f[7] + f[6] + f[5] - f[0] * 2.) * 2.;
    /*R[1] = f[8] + f[7] + f[6] + f[5] + (-f[4] - f[3] - f[2] - f[1] + f[0] * 2.) * 2.;
    R[2] = f[8] - f[7] - f[6] + f[5] + (f[3] - f[1]) * 2.;
    R[3] = -f[8] - f[7] + f[6] + f[5] + (f[4] - f[2]) * 2.;*/
    R[4] = -f[4] + f[3] - f[2] + f[1];
    R[5] = -f[8] + f[7] - f[6] + f[5];
    usq = u[1] * u[1] + u[0] * u[0];

    R[0] = R[0] - (-2. * d + 3. * usq);
    /*R[1] = R[1] - (d - 3. * usq);
    R[2] = R[2] - (-u[0]);
    R[3] = R[3] - (-u[1]);*/
    R[4] = R[4] - (u[0] * u[0] - u[1] * u[1]);
    R[5] = R[5] - (u[0] * u[1]);
	real_t Qxy=0;
	Qxy += (-0.02*(3*omega)/2)*R[5];
	return Qxy;
}

CudaDeviceFunction real_t getQyy()
{
    real_t u[2], usq, d, R[6], uf;
    d = f[8] + f[7] + f[6] + f[5] + f[4] + f[3] + f[2] + f[1] + f[0];
    u[0] = f[8] - f[7] - f[6] + f[5] - f[3] + f[1];
    u[1] = -f[8] - f[7] + f[6] + f[5] - f[4] + f[2];
    R[0] = -f[4] - f[3] - f[2] - f[1] + (f[8] + f[7] + f[6] + f[5] - f[0] * 2.) * 2.;
    /*R[1] = f[8] + f[7] + f[6] + f[5] + (-f[4] - f[3] - f[2] - f[1] + f[0] * 2.) * 2.;
    R[2] = f[8] - f[7] - f[6] + f[5] + (f[3] - f[1]) * 2.;
    R[3] = -f[8] - f[7] + f[6] + f[5] + (f[4] - f[2]) * 2.;*/
    R[4] = -f[4] + f[3] - f[2] + f[1];
    R[5] = -f[8] + f[7] - f[6] + f[5];
    usq = u[1] * u[1] + u[0] * u[0];

    R[0] = R[0] - (-2. * d + 3. * usq);
    /*R[1] = R[1] - (d - 3. * usq);
    R[2] = R[2] - (-u[0]);
    R[3] = R[3] - (-u[1]);*/
    R[4] = R[4] - (u[0] * u[0] - u[1] * u[1]);
    R[5] = R[5] - (u[0] * u[1]);
	real_t Qyy=0;
	Qyy += (-0.02*(3*omega)/2)*(R[0]/6-R[4]/2);
	return Qyy;
}

CudaDeviceFunction real_t getSS()
{
    real_t u[2], usq, d, R[6], uf;
    d = f[8] + f[7] + f[6] + f[5] + f[4] + f[3] + f[2] + f[1] + f[0];
    u[0] = f[8] - f[7] - f[6] + f[5] - f[3] + f[1];
    u[1] = -f[8] - f[7] + f[6] + f[5] - f[4] + f[2];
    R[0] = -f[4] - f[3] - f[2] - f[1] + (f[8] + f[7] + f[6] + f[5] - f[0] * 2.) * 2.;
    /*R[1] = f[8] + f[7] + f[6] + f[5] + (-f[4] - f[3] - f[2] - f[1] + f[0] * 2.) * 2.;
    R[2] = f[8] - f[7] - f[6] + f[5] + (f[3] - f[1]) * 2.;
    R[3] = -f[8] - f[7] + f[6] + f[5] + (f[4] - f[2]) * 2.;*/
    R[4] = -f[4] + f[3] - f[2] + f[1];
    R[5] = -f[8] + f[7] - f[6] + f[5];
    usq = u[1] * u[1] + u[0] * u[0];

    R[0] = R[0] - (-2. * d + 3. * usq);
    /*R[1] = R[1] - (d - 3. * usq);
    R[2] = R[2] - (-u[0]);
    R[3] = R[3] - (-u[1]);*/
    R[4] = R[4] - (u[0] * u[0] - u[1] * u[1]);
    R[5] = R[5] - (u[0] * u[1]);
	real_t Qxx, Qxy, Qyy;
	real_t SS=0;
	Qxx = (-0.02*(3*omega)/2)*(R[0]/6+R[4]/2);
	Qxy = (-0.02*(3*omega)/2)*R[5];
	Qyy = (-0.02*(3*omega)/2)*(R[0]/6-R[4]/2);
	SS=sqrt((Qxx*Qxx+Qyy*Qyy)/3-(Qxx*Qyy)/3+Qxy*Qxy);
	return SS;
}

CudaDeviceFunction vector_t getU()
{
    real_t d = f[8] + f[7] + f[6] + f[5] + f[4] + f[3] + f[2] + f[1] + f[0];
    vector_t u;
    u.x = f[8] - f[7] - f[6] + f[5] - f[3] + f[1];
    u.y = -f[8] - f[7] + f[6] + f[5] - f[4] + f[2];

//      u.x /= d;
//      u.y /= d;
    u.z = 0.0;
    return u;
}

CudaDeviceFunction float2 Color()
{
    float2 ret;
//        vector_t u = getU();
//        ret.x = sqrt(u.x*u.x + u.y*u.y);

    ret.x = (T[8] + T[7] + T[6] + T[5] + T[4] + T[3] + T[2] + T[1] + T[0]);
    ret.x = ret.x / 111;
//      ret.x = wb;
    if (NodeType == NODE_Solid)
	ret.y = 0;
    else
	ret.y = 1;
    return ret;
}

CudaDeviceFunction void BounceBack()
{
    real_t uf;
#define dump 1
    uf = f[3];
    f[3] = f[1];
    f[1] = uf;
    uf = T[3];
    T[3] = dump * T[1];
    T[1] = dump * uf;
    uf = f[4];
    f[4] = f[2];
    f[2] = uf;
    uf = T[4];
    T[4] = dump * T[2];
    T[2] = dump * uf;
    uf = f[7];
    f[7] = f[5];
    f[5] = uf;
    uf = T[7];
    T[7] = dump * T[5];
    T[5] = dump * uf;
    uf = f[8];
    f[8] = f[6];
    f[6] = uf;
    uf = T[8];
    T[8] = dump * T[6];
    T[6] = dump * uf;

}

// 0 1 2 3 4 5 6 7 8
// 1 5 2 6 3 7 4 8 0

CudaDeviceFunction void EVelocity()
{
    real_t rho, ru;
    const real_t ux0 = InletVelocity;
    rho = (f[0] + f[2] + f[4] + 2. * (f[1] + f[5] + f[8])) / (1. + ux0);
    ru = rho * ux0;
    f[3] = f[1] - (2. / 3.) * ru;
    f[7] = f[5] - (1. / 6.) * ru + (1. / 2.) * (f[2] - f[4]);
    f[6] = f[8] - (1. / 6.) * ru + (1. / 2.) * (f[4] - f[2]);
}

CudaDeviceFunction void WPressure()
{
    real_t ru, ux0;
    real_t rho = InletDensity;
    ux0 = -1. + (f[0] + f[2] + f[4] + 2. * (f[3] + f[7] + f[6])) / rho;
    ru = rho * ux0;

    f[1] = f[3] - (2. / 3.) * ru;
    f[5] = f[7] - (1. / 6.) * ru + (1. / 2.) * (f[4] - f[2]);
    f[8] = f[6] - (1. / 6.) * ru + (1. / 2.) * (f[2] - f[4]);
#define rho_bar (InletTemperature)
    rho = 6. * (rho_bar - (T[0] + T[2] + T[4] + T[3] + T[7] + T[6]));
    T[1] = (1. / 9.) * rho;
    T[5] = (1. / 36.) * rho;
    T[8] = (1. / 36.) * rho;
}

CudaDeviceFunction void eqWVelocity()
{
    real_t rho;
    const real_t u[2] = { InletVelocity, 0. };
    rho = (f[0] + f[2] + f[4] + 2. * (f[3] + f[7] + f[6])) / (1. - u[0]);
    SetEquilibrum(rho, u);
}

CudaDeviceFunction void WVelocity()
{
    real_t rho, ru;
    const real_t u[2] = { InletVelocity, 0. };
    rho = (f[0] + f[2] + f[4] + 2. * (f[3] + f[7] + f[6])) / (1. - u[0]);
    ru = rho * u[0];
    f[1] = f[3] + (2. / 3.) * ru;
    f[5] = f[7] + (1. / 6.) * ru + (1. / 2.) * (f[4] - f[2]);
    f[8] = f[6] + (1. / 6.) * ru + (1. / 2.) * (f[2] - f[4]);
#define rho_bar (InletTemperature)
    rho = 6. * (rho_bar - (T[0] + T[2] + T[4] + T[3] + T[7] + T[6]));
    T[1] = (1. / 9.) * rho;
    T[5] = (1. / 36.) * rho;
    T[8] = (1. / 36.) * rho;

}

CudaDeviceFunction void EPressure()
{
    real_t ru, ux0;
    real_t rho = 1.0;
    ux0 = -1. + (f[0] + f[2] + f[4] + 2. * (f[1] + f[5] + f[8])) / rho;
    ru = rho * ux0;

    f[3] = f[1] - (2. / 3.) * ru;
    f[7] = f[5] - (1. / 6.) * ru + (1. / 2.) * (f[2] - f[4]);
    f[6] = f[8] - (1. / 6.) * ru + (1. / 2.) * (f[4] - f[2]);

    rho = 6. * (T[1] + T[5] + T[8]);	///(1-3*ux0);
    T[3] = (1. / 9.) * rho;
    T[7] = (1. / 36.) * rho;
    T[6] = (1. / 36.) * rho;
}

CudaDeviceFunction void Run()
{
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
    if ((NodeType & NODE_MRT)) {
	CollisionMRT();
    }
}

CudaDeviceFunction void SetEquilibrum(const real_t d, const real_t u[2])
{
    real_t usq, uf;
    usq = (u[1] * u[1] + u[0] * u[0]) * 3.;

//-- 1 -------------------------------------------------
    uf = 0;
    uf = 1 + uf + (-usq + uf * uf) / 2.;
    uf = uf * d;
    f[0] = uf * 4. / 9.;
//-- 2 -------------------------------------------------
    uf = u[0] * 3.;
    uf = 1 + uf + (-usq + uf * uf) / 2.;
    uf = uf * d;
    f[1] = uf / 9.;
//-- 3 -------------------------------------------------
    uf = u[1] * 3.;
    uf = 1 + uf + (-usq + uf * uf) / 2.;
    uf = uf * d;
    f[2] = uf / 9.;
//-- 4 -------------------------------------------------
    uf = -u[0] * 3.;
    uf = 1 + uf + (-usq + uf * uf) / 2.;
    uf = uf * d;
    f[3] = uf / 9.;
//-- 5 -------------------------------------------------
    uf = -u[1] * 3.;
    uf = 1 + uf + (-usq + uf * uf) / 2.;
    uf = uf * d;
    f[4] = uf / 9.;
//-- 6 -------------------------------------------------
    uf = (u[1] + u[0]) * 3.;
    uf = 1 + uf + (-usq + uf * uf) / 2.;
    uf = uf * d;
    f[5] = uf / 36.;
//-- 7 -------------------------------------------------
    uf = (u[1] - u[0]) * 3.;
    uf = 1 + uf + (-usq + uf * uf) / 2.;
    uf = uf * d;
    f[6] = uf / 36.;
//-- 8 -------------------------------------------------
    uf = (-u[1] - u[0]) * 3.;
    uf = 1 + uf + (-usq + uf * uf) / 2.;
    uf = uf * d;
    f[7] = uf / 36.;
//-- 9 -------------------------------------------------
    uf = (-u[1] + u[0]) * 3.;
    uf = 1 + uf + (-usq + uf * uf) / 2.;
    uf = uf * d;
    f[8] = uf / 36.;

}

CudaDeviceFunction void Init()
{
    real_t u[2] = { InletVelocity, 0. };
    real_t d = 1.0;
    SetEquilibrum(d, u);
//      for (int i =0; i<9;i++) T[i] = 0;

    T[0] = InitTemperature * 0.444444;

    T[1] = InitTemperature * 0.111111;

    T[2] = InitTemperature * 0.111111;

    T[3] = InitTemperature * 0.111111;

    T[4] = InitTemperature * 0.111111;

    T[5] = InitTemperature * 0.027778;

    T[6] = InitTemperature * 0.027778;

    T[7] = InitTemperature * 0.027778;

    T[8] = InitTemperature * 0.027778;

}

CudaDeviceFunction void CollisionMRT()
{
    real_t u[2], usq, d, R[6], uf;
    d = f[8] + f[7] + f[6] + f[5] + f[4] + f[3] + f[2] + f[1] + f[0];
    u[0] = f[8] - f[7] - f[6] + f[5] - f[3] + f[1];
    u[1] = -f[8] - f[7] + f[6] + f[5] - f[4] + f[2];
    R[0] = -f[4] - f[3] - f[2] - f[1] + (f[8] + f[7] + f[6] + f[5] - f[0] * 2.) * 2.;
    R[1] = f[8] + f[7] + f[6] + f[5] + (-f[4] - f[3] - f[2] - f[1] + f[0] * 2.) * 2.;
    R[2] = f[8] - f[7] - f[6] + f[5] + (f[3] - f[1]) * 2.;
    R[3] = -f[8] - f[7] + f[6] + f[5] + (f[4] - f[2]) * 2.;
    R[4] = -f[4] + f[3] - f[2] + f[1];
    R[5] = -f[8] + f[7] - f[6] + f[5];
    usq = u[1] * u[1] + u[0] * u[0];

	R[0] -= (-2. * d + 3. * usq);
    R[1] -= (d - 3. * usq);
    R[2] -= (-u[0]);
    R[3] -= (-u[1]);
    R[4] -= (u[0] * u[0] - u[1] * u[1]);
    R[5] -= (u[0] * u[1]);

	real_t Qxx, Qxy, Qyy;
	real_t SS=0;
	Qxx = (-0.02*(3*omega)/2)*(R[0]/6+R[4]/2);
	Qxy = (-0.02*(3*omega)/2)*R[5];
	Qyy = (-0.02*(3*omega)/2)*(R[0]/6-R[4]/2);
	SS=sqrt((Qxx*Qxx+Qyy*Qyy)/3-(Qxx*Qyy)/3+Qxy*Qxy);

	R[0] = R[0] * (1 - S2) + (-2. * d + 3. * usq);
    R[1] = R[1] * (1 - S3) + (d - 3. * usq);
    R[2] = R[2] * (1 - S5) + (-u[0]);
    R[3] = R[3] * (1 - S7) + (-u[1]);
    R[4] = R[4] * (1 - S8) + (u[0] * u[0] - u[1] * u[1]);
    R[5] = R[5] * (1 - S9) + (u[0] * u[1]);

    //R[0] = R[0] * (1 - S2) + S2 * (-2. * d + 3. * usq);
    //R[1] = R[1] * (1 - S3) + S3 * (d - 3. * usq);
    //R[2] = R[2] * (1 - S5) + S5 * (-u[0]);
    //R[3] = R[3] * (1 - S7) + S7 * (-u[1]);
    //R[4] = R[4] * (1 - S8) + S8 * (u[0] * u[0] - u[1] * u[1]);
    //R[5] = R[5] * (1 - S9) + S9 * (u[0] * u[1]);

    f[0] = (R[1] - R[0] + d) / 9.;
    f[1] = (-R[0] + R[4] * 9. + (-R[1] + d * 2. + (-R[2] + u[0]) * 3.) * 2.) / 36.;
    f[2] = (-R[0] - R[4] * 9. + (-R[1] + d * 2. + (-R[3] + u[1]) * 3.) * 2.) / 36.;
    f[3] = (-R[0] + R[4] * 9. + (-R[1] + d * 2. + (R[2] - u[0]) * 3.) * 2.) / 36.;
    f[4] = (-R[0] - R[4] * 9. + (-R[1] + d * 2. + (R[3] - u[1]) * 3.) * 2.) / 36.;
    f[5] = (R[1] + (R[0] + d * 2.) * 2. + (R[3] + R[2] + R[5] * 3. + (u[1] + u[0]) * 2.) * 3.) / 36.;
    f[6] = (R[1] + (R[0] + d * 2.) * 2. + (R[3] - R[2] - R[5] * 3. + (u[1] - u[0]) * 2.) * 3.) / 36.;
    f[7] = (R[1] + (R[0] + d * 2.) * 2. + (-R[3] - R[2] + R[5] * 3. + (-u[1] - u[0]) * 2.) * 3.) / 36.;
    f[8] = (R[1] + (R[0] + d * 2.) * 2. + (-R[3] + R[2] - R[5] * 3. + (-u[1] + u[0]) * 2.) * 3.) / 36.;

    real_t us[2];
    us[0] = u[0] / d;
    us[1] = u[1] / d;
    d = T[8] + T[7] + T[6] + T[5] + T[4] + T[3] + T[2] + T[1] + T[0];
    u[0] = T[8] - T[7] - T[6] + T[5] - T[3] + T[1];
    u[1] = -T[8] - T[7] + T[6] + T[5] - T[4] + T[2];
    R[0] = -T[4] - T[3] - T[2] - T[1] + (T[8] + T[7] + T[6] + T[5] - T[0] * 2.) * 2.;
    R[1] = T[8] + T[7] + T[6] + T[5] + (-T[4] - T[3] - T[2] - T[1] + T[0] * 2.) * 2.;
    R[2] = T[8] - T[7] - T[6] + T[5] + (T[3] - T[1]) * 2.;
    R[3] = -T[8] - T[7] + T[6] + T[5] + (T[4] - T[2]) * 2.;
    R[4] = -T[4] + T[3] - T[2] + T[1];
    R[5] = -T[8] + T[7] - T[6] + T[5];
	
	if ((NodeType & NODE_ADDITIONALS) == NODE_Outlet2)
	{
		AddToDestroyedCellFlux(d*us[0]);
		AddToOutFlux(us[0]);
	}
    if ((NodeType & NODE_ADDITIONALS) == NODE_Heater)
	d = 100;

#define Tom omegaT
#define Tom2 omegaT

    real_t omegaT = FluidAlfa;
    omegaT = 1.0 / (3 * omegaT + 0.5);

	R[0] -= (-2 * d);
    R[1] -= (d);
    R[2] -= (-us[0] * d);
    R[3] -= (-us[1] * d);
    R[4] -= 0;
    R[5] -= 0;
    u[0] -= (us[0] * d);
    u[1] -= (us[1] * d);

	if ((NodeType & NODE_ADDITIONALS) == NODE_Destroy)
	{
		real_t dch;
		dch=DestructionRate*pow(SS,DestructionPower);
		d=d+(1-d)*dch;
	}


	R[0] = R[0] * (1 - Tom2) + (-2 * d);
    R[1] = R[1] * (1 - Tom2) + (d);
    R[2] = R[2] * (1 - Tom2) + (-us[0] * d);
    R[3] = R[3] * (1 - Tom2) + (-us[1] * d);
    R[4] = R[4] * (1 - Tom);
    R[5] = R[5] * (1 - Tom);
    u[0] = u[0] * (1 - Tom2) + (us[0] * d);
    u[1] = u[1] * (1 - Tom2) + (us[1] * d);

    //R[0] = R[0] * (1 - Tom2) + (-2 * d) * Tom2;
    //R[1] = R[1] * (1 - Tom2) + (d) * Tom2;
    //R[2] = R[2] * (1 - Tom2) + (-us[0] * d) * Tom2;
    //R[3] = R[3] * (1 - Tom2) + (-us[1] * d) * Tom2;
    //R[4] = R[4] * (1 - Tom);
    //R[5] = R[5] * (1 - Tom);
    //u[0] = u[0] * (1 - Tom2) + (us[0] * d) * Tom2;
    //u[1] = u[1] * (1 - Tom2) + (us[1] * d) * Tom2;

    T[0] = (R[1] - R[0] + d) / 9.;
    T[1] = (-R[0] + R[4] * 9. + (-R[1] + d * 2. + (-R[2] + u[0]) * 3.) * 2.) / 36.;
    T[2] = (-R[0] - R[4] * 9. + (-R[1] + d * 2. + (-R[3] + u[1]) * 3.) * 2.) / 36.;
    T[3] = (-R[0] + R[4] * 9. + (-R[1] + d * 2. + (R[2] - u[0]) * 3.) * 2.) / 36.;
    T[4] = (-R[0] - R[4] * 9. + (-R[1] + d * 2. + (R[3] - u[1]) * 3.) * 2.) / 36.;
    T[5] = (R[1] + (R[0] + d * 2.) * 2. + (R[3] + R[2] + R[5] * 3. + (u[1] + u[0]) * 2.) * 3.) / 36.;
    T[6] = (R[1] + (R[0] + d * 2.) * 2. + (R[3] - R[2] - R[5] * 3. + (u[1] - u[0]) * 2.) * 3.) / 36.;
    T[7] = (R[1] + (R[0] + d * 2.) * 2. + (-R[3] - R[2] + R[5] * 3. + (-u[1] - u[0]) * 2.) * 3.) / 36.;
    T[8] = (R[1] + (R[0] + d * 2.) * 2. + (-R[3] + R[2] - R[5] * 3. + (-u[1] + u[0]) * 2.) * 3.) / 36.;

	


//    if ((NodeType & NODE_OBJECTIVE) == NODE_Outlet) AddToObjective(d*us[0]);

}
