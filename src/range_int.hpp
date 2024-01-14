#ifndef RANGE_INT_HPP
#define RANGE_INT_HPP

#include "cross.h"

template <int A, int B=0, int C=A, int D=B>
struct range_int {
  const int val;
  CudaDeviceFunction range_int(const int& val_=A) : val(val_) {}
  CudaDeviceFunction operator int () const { return val; }
  template <int A_, int B_, int C_, int D_>
  CudaDeviceFunction range_int<A+A_, B+B_, C+C_, D+D_> operator + (const range_int<A_, B_, C_, D_>& other) const {
    return val + other.val;
  }
  template <int A_, int B_, int C_, int D_>
  CudaDeviceFunction range_int<A-C_, B-D_, C-A_, D-B_> operator - (const range_int<A_, B_, C_, D_>& other) const {
    return val - other.val;
  }
  CudaDeviceFunction range_int< -C, -D, -A, -B> operator - () const {
    return -val;
  }
  template <int A_, int B_, int C_, int D_>
  CudaDeviceFunction bool operator <  (const range_int<A_, B_, C_, D_>& other) const {
    if (D <  B_) return true;
    if (D == B_) if (C <  A_) return true;
    if (B >  D_) return false;
    if (B == D_) if (A >= C_) return false;
    return val <  other.val;
  }
  template <int A_, int B_, int C_, int D_>
  CudaDeviceFunction bool operator <= (const range_int<A_, B_, C_, D_>& other) const {
    if (D <  B_) return true;
    if (D == B_) if (C <=  A_) return true;
    if (B >  D_) return false;
    if (B == D_) if (A > C_) return false;
    return val <= other.val;
  }
  template <int A_, int B_, int C_, int D_>
  CudaDeviceFunction bool operator >  (const range_int<A_, B_, C_, D_>& other) const {
    if (D >  B_) return true;
    if (D == B_) if (C >  A_) return true;
    if (B <  D_) return false;
    if (B == D_) if (A <= C_) return false;
    return val >  other.val;
  }
  template <int A_, int B_, int C_, int D_>
  CudaDeviceFunction bool operator >= (const range_int<A_, B_, C_, D_>& other) const {
    if (D >  B_) return true;
    if (D == B_) if (C >=  A_) return true;
    if (B <  D_) return false;
    if (B == D_) if (A < C_) return false;
    return val >= other.val;
  }
  template <int A_, int B_, int C_, int D_>
  CudaDeviceFunction static range_int<A_, B_, C_, D_> ensure (const range_int<A_, B_, C_, D_>& x) {
    static_assert( A <= A_ , "range_int range error");
    static_assert( B <= B_ , "range_int range error");
    static_assert( C >= C_ , "range_int range error");
    static_assert( D >= D_ , "range_int range error");
    return x;
  }
  CudaDeviceFunction static range_int<A, B, C, D> ensure (const int& x) {
    return x;
  }
  
};

template <int A, int B, int C, int D>
CudaDeviceFunction range_int<A, B, C, D> ensure_range_int (const range_int<A, B, C, D>& x) {
  return x;
}

template <int A, int B, int C, int D>
void print_range_int (const range_int<A, B, C, D>& x) {
  printf("%d+N*%d <= %d <= %d+N*%d\n",A,B,x.val,C,D);
}


#endif
