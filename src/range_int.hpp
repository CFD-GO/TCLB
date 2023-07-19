#ifndef RANGE_INT_HPP
#define RANGE_INT_HPP

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
};

#endif
