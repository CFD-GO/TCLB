// Model for solving the wave equation as a system of first order DE's
// u'' = lap(u)
// u'' = c(u_xx + u_yy)


CudaDeviceFunction float2 Color() {
  float2 ret;
  ret.x = getU();
  ret.y = 1;
  return ret;
}

CudaDeviceFunction real_t getU() {
    return u(0,0);
  }

CudaDeviceFunction void Init() {
    u = Value;
    v = 0;
 }

CudaDeviceFunction void Run() { 
  real_t lap_u = u(-1,0) + u(1,0) + u(0,-1) + u(0,1) - 4*u(0,0);
  real_t lap_v = v(-1,0) + v(1,0) + v(0,-1) + v(0,1) - 4*v(0,0);
  real_t a = Speed * Speed * lap_u + Viscosity * lap_v;
  v = v(0,0) + a;
  u = u(0,0) + v;

  if ((NodeType & NODE_BOUNDARY) == NODE_Dirichlet)  {
    u = Value;
    v = 0;
  }
}