
#ifndef REGION_H
#include <stdio.h>
class lbRegion {
public:
  int dx,dy,dz;
  int nx,ny,nz;
  inline lbRegion():dx(0),dy(0),dz(0),nx(1),ny(1),nz(1){}
  inline lbRegion(int w, int h):dx(0),dy(0),dz(0),nx(w),ny(h),nz(1) {};
  inline lbRegion(int x, int y, int w, int h):dx(x),dy(y),dz(0),nx(w),ny(h),nz(1){};
  inline lbRegion(int x, int y, int z, int w, int h, int d):dx(x),dy(y),dz(z),nx(w),ny(h),nz(d){};
  CudaHostFunction CudaDeviceFunction inline int size() { return nx*ny*nz; };
  CudaHostFunction CudaDeviceFunction inline size_t sizeL() { return nx*ny*nz; };
  inline int isIn(int x, int y) { return (x >= dx) && (y >= dy) && (x-dx < nx) && (y-dy < ny); }
  inline int isIn(int x, int y, int z) { return (x >= dx) && (y >= dy) && (z >= dz) && (x-dx < nx) && (y-dy < ny) && (z-dz < nz); }
  inline int isEqual(const lbRegion& other) { return (other.dx == dx) && (other.dy == dy) && (other.dz == dz) && (other.nx == nx) && (other.ny == ny) && (other.nz == nz); }
  inline lbRegion intersect(lbRegion w) {
    lbRegion ret;
    ret.dx = max(dx,w.dx);
    ret.dy = max(dy,w.dy);
    ret.dz = max(dz,w.dz);
    ret.nx = min(dx+nx , w.dx+w.nx) - ret.dx;
    ret.ny = min(dy+ny,w.dy+w.ny) - ret.dy;
    ret.nz = min(dz+nz,w.dz+w.nz) - ret.dz;
    if (ret.nx <= 0 || ret.ny <= 0 || ret.nz <= 0) { ret.nx = ret.ny = ret.nz = 0; };
    return ret;
  };
  inline int offset(int x,int y) {
    return (x-dx) + (y-dy) * nx;
  };
  inline void print() {
    printf("Region: %dx%dx%d + %d,%d,%d\n", nx,ny,nz,dx,dy,dz);
  };
  CudaHostFunction CudaDeviceFunction inline int offset(int x,int y,int z) {
    return (x-dx) + (y-dy) * nx + (z-dz) * nx * ny;
  };
  CudaHostFunction CudaDeviceFunction inline size_t offsetL(int x,int y,int z) {
    return (x-dx) + (y-dy) * nx + (z-dz) * nx * ny;
  };

};

#endif
#define REGION_H 1
