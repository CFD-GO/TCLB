#ifndef SPLINE_H

#include <vector>

// B-Spline knot
// repeat 0 k-times
// linear from k to n
// repeat 1 k-times
inline double knot_bs(int i, int n, int k) {
  if (i < k+1) return 0;
  if (i < n) return 1.0*(i-k)/(n-k);
  return 1;
}

// 
template <typename T>
T bspline_mod(double x, std::vector<T>& p, int k) {
  int n = p.size();
  int i = floor(x * (n - k))+k; // [k, n-1]
  if (k > n-1) {
    ERROR("Wrong order in bspline!");
    k = n-1;
  }
  if (i > n-1) i = n-1;
  if (i < k) i = k;
//  knot_bs(i,n,k) < x < knot_bs(i+1,n,k)
  for (int j=k; j>0; j--) {
    for (int l=0; l<j; l++) {
      double a = (x - knot_bs(i-l,n,k)) / (knot_bs(i-l+j,n,k) - knot_bs(i-l,n,k));
      p[i-l] = a*p[i-l] + (1-a)*p[i-l-1];
    }
  }
  return p[i];
}

template <typename T>
T bspline(double x, const std::vector<T>& p, int k) {
  static std::vector<T> pcopy;
  size_t n = p.size();
  if (pcopy.size() != n) pcopy.resize(n);
  for (size_t i=0;i<n;i++) pcopy[i] = p[i];
  return bspline_mod(x,pcopy,k);
}
#define SPLINE_H
#endif
