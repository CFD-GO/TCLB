#ifndef SPLINE_H

#include <vector>

// B-Spline knot
// repeat 0 k-times
// linear from k to n
// repeat 1 k-times
inline double knot_bs(int i, int n, int k,bool cut) {
  if (! cut) return 1.0*(i-k)/(n-k);
  if (i < k+1) return 0;
  if (i < n) return 1.0*(i-k)/(n-k);
  return 1;
}

template <typename T>
T bspline_mod(double x, std::vector<T>& p, int k,bool cut) {
  int n = p.size();
  int i = floor(x * (n - k))+k; // [k, n-1]
  if (k > n-1) {
    
    k = n-1;
  }
  if (i > n-1) i = n-1;
  if (i < k) i = k;
  //  knot_bs(i,n,k) < x < knot_bs(i+1,n,k)
  for (int j=k; j>0; j--) {
    for (int l=0; l<j; l++) {
      double a = (x - knot_bs(i-l,n,k,cut)) / (knot_bs(i-l+j,n,k,cut) - knot_bs(i-l,n,k,cut));
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
  return bspline_mod(x,pcopy,k,true);
}

inline double bspline_b(double x, int n, int w, int k, bool per) {
  static std::vector<double> pcopy;
  if (w >= n) w=n-1;
  if (w < 0) w=0;
  if (per) n = n + k;
  if (pcopy.size() != (size_t) n) pcopy.resize(n);
  for (int i=0;i<n;i++) pcopy[i] = 0;
  pcopy[w] = 1;
  w = w + n - k;
  if (per && w < n) pcopy[w] = 1;
  return bspline_mod(x,pcopy,k,! per);
}

#define SPLINE_H
#endif
