#ifndef REGION_H
#define REGION_H 1

#include "cross.h"

#include <cstdio>

struct lbRegion {
  int dx = 0, dy = 0, dz = 0;
  int nx = 1, ny = 1, nz = 1;
  lbRegion() = default;
  lbRegion(int w, int h) : nx(w), ny(h) {}
  lbRegion(int x, int y, int w, int h) : dx(x), dy(y), nx(w), ny(h) {}
  lbRegion(int x, int y, int z, int w, int h, int d) : dx(x), dy(y), dz(z), nx(w), ny(h), nz(d) {}
  CudaHostFunction CudaDeviceFunction int size() const { return nx*ny*nz; }
  CudaHostFunction CudaDeviceFunction size_t sizeL() const { return nx*ny*nz; }
  int isIn(int x, int y) const { return (x >= dx) && (y >= dy) && (x-dx < nx) && (y-dy < ny); }
  int isIn(int x, int y, int z) const { return (x >= dx) && (y >= dy) && (z >= dz) && (x-dx < nx) && (y-dy < ny) && (z-dz < nz); }
  int isEqual(const lbRegion& other) const { return (other.dx == dx) && (other.dy == dy) && (other.dz == dz) && (other.nx == nx) && (other.ny == ny) && (other.nz == nz); }
  lbRegion intersect(const lbRegion& w) const {
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
  int offset(int x,int y) const {
    return (x-dx) + (y-dy) * nx;
  };
  void print() const {
    printf("Region: %dx%dx%d + %d,%d,%d\n", nx,ny,nz,dx,dy,dz);
  };
  CudaHostFunction CudaDeviceFunction int offset(int x,int y,int z) const {
    return (x-dx) + (y-dy) * nx + (z-dz) * nx * ny;
  };
  CudaHostFunction CudaDeviceFunction size_t offsetL(int x,int y,int z) const {
    return (x-dx) + (y-dy) * nx + (z-dz) * nx * ny;
  };
};

#endif
