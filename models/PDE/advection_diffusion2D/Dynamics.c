// Model for solving the advection-diffusion equation:
// phi' = -U * grad(phi) + c*lap(phi)
// phi' = -[ux,uy]' * [phi_x,phi_y] + c * (phi_xx + phi_yy)

CudaDeviceFunction float2 Color() {
  float2 ret;
  ret.x = getPhi();
  ret.y = 1;
  return ret;
}

CudaDeviceFunction real_t getPhi() {
    return phi(0,0);
  }

CudaDeviceFunction void Init() {
    phi = Value;
 }

CudaDeviceFunction void Run() { 
  real_t lap_phi = phi(-1,0) + phi(1,0) + phi(0,-1) + phi(0,1) - 4*phi(0,0);
  real_t grad_phi[2];
  grad_phi[0] = (phi(1,0)-phi(-1,0))/2;
  grad_phi[1] = (phi(0,1)-phi(0,-1))/2;
    
  real_t temp = diff_coeff * diff_coeff * lap_phi - ux*grad_phi[0] - uy*grad_phi[1];
  phi = phi(0,0) + temp;

  if ((NodeType & NODE_BOUNDARY) == NODE_Dirichlet)  {
    phi = Value;
  }
}