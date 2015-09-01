q_c = function(x) {
  x*c(1,-1,-1,-1)
}

q_m = (function() {
  A = matrix(1,4,4)
  A[1,1:4]=1:4
  A[1:4,1]=1:4
  A[2:4,2:4]=c(1,4,3,4,1,2,3,2,1)
  B = matrix(1,4,4)
  B[2:4,2:4]=c(-1, 1,-1,-1,-1, 1, 1,-1,-1)
  
  function(x,y) {
    ret = PV(rep(0,4))
    for (i in 1:4) {
      for (j in 1:4) {
        k = A[i,j]
        ret[k] = ret[k] + x[i] * y[j] * B[i,j]
      }
    }
    ret
  }
})()

q_rot = function(p,q,np,inv=FALSE) {
  p = V(0,p)
  if (inv) {
    p_ = q_m(q_c(q),q_m(p,q))
  } else {
    p_ = q_m(q_m(q,p),q_c(q))
  }
  z1 = sum(q*q) - 1
  h  = PV("h")
  h1 = sum(q*p)
  p_ = p_ - 2*(h1-h)*q + z1*p
  cat("real_t h;\n")
  C(h,h1)
  C(np,p_[2:4])
}
