library(polyAlgebra)
library(Matrix)

m = t(matrix(c(1,1,1,0,1,-1,0,1,1),3,3))
I = diag(3)
lum = expand(lu(m))
if (any(lum$P != diag(3))) stop("wrong pivot")
FMT_mat = as.matrix(I - solve(lum$L) + lum$U)
lum = expand(lu(solve(m)))
if (any(lum$P != diag(3))) stop("wrong pivot")
FMT_inv = as.matrix(I - solve(lum$L) + lum$U)

FMT = function(I,M,sel,inverse=FALSE) {
  if (inverse) { mat = FMT_inv } else { mat = FMT_mat }
  if (missing(sel)) sel = rep(TRUE,nrow(I))
  for (s in 1:ncol(I)) {
    ret = by(1:nrow(I),I[,-s],function(i) if (all(sel[i])) {
      C(M[i], mat %*% M[i],sep = "; ")
      cat("\n")
    })
  }
}

FMT_num = list(
  "1" = c(1,2,3),
  "2" = c(1,2,4,3,6,7,5,9,8),
  "3" = c(1,2,3,4,8,9,5,10,11,6,12,13,16,20,21,17,22,23,7,14,15,18,24,25,19,26,27)
)
