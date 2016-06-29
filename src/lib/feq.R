#library(polyAlgebra)

gcd <- function(a,b) ifelse (b==0, a, gcd(b, a %% b))
library(numbers)

MRT_polyMatrix = function(U) {
  if (any(U >  1)) stop("Too high velocities in calculate_feq")
  if (any(U < -1)) stop("Too high velocities in calculate_feq")
  d = ncol(U)
  d2 = expand.grid(i=1:d,j=1:d)
  D2 = NULL; for (i in 1:nrow(d2)) { D2 = cbind(D2, U[,d2$i[i]] * U[,d2$j[i]]) }
  dim(D2) = c(nrow(U), d,d)
  p = ifelse(U < 0,2,U)
  p = p[order(rowSums(p)),]
  W = NULL; for (i in 1:nrow(p)) {W = cbind(W, apply(t(U) ^ p[i,],2,prod)) }
  list(order=rowSums(p), mat=W, p=p, D2=D2)
}

MRT_integerOrtogonal = function(M) {
  for (i in 2:ncol(M))
  {
    a = as.integer(t(M[,1:(i-1)]) %*% M[,i])
    b = as.integer(colSums(M[,1:(i-1),drop=FALSE]**2))
    g = gcd(a,b)
    a = a/g
    b = b/g
    if (length(b) == 1) l = b else l=mLCM(b);
    M[,i] = M[,i]*l - M[,1:(i-1),drop=F] %*% (l*a/b)
  }
  M
}

MRT_eq = function(U, rho=PV("rho"), J=PV(c("Jx","Jy","Jz")), sigma2=1/3, order=2, ortogonal=TRUE, mat, correction) {
  rho_str = ToC(rho)
  W = MRT_polyMatrix(U)
  p = W$p
  H = rho[rep(1,nrow(U))];
  for (j in 1:nrow(U))
  {
    for (i in 1:ncol(U))
    {
      if (p[j,i] == 1) H[j] = H[j] * J[i] * rho^(-1)
      if (p[j,i] == 2) H[j] = H[j] * (J[i]^2 * rho^(-2) + sigma2)
    }
  }
  H = gapply(H, function(x) if (inherits(x,"pAlg")) {
	i=names(x) %in% c(".M",rho_str);
	h=rowSums(abs(x[,!i,drop=FALSE]));
	sel = h <= order;
	x[sel,] 
    } else {
	x
  })
  ret = list(Req=H, mat=W$mat, p=W$p, order=W$order, U=U, D2=W$D2)
  if (!missing(correction)) {
	sel = ret$order > 3
	if (sum(sel) != length(correction)) stop("Correction of wrong length in MRT_eq")
	ret$Req[sel] = ret$Req[sel] + correction
  }
  if (missing(mat)) {
	mat = attr(U,"MAT")
  }
  if (! is.null(mat)) {
	if (! is.matrix(mat)) stop("\"mat\" provided to MRT_eq is not a matrix")
	M = mat
	ret$order  = apply(abs(solve(W$mat) %*% M) > 1e-10,2,function(x) max(W$order[x]))
	ret$Req = ret$Req %*% (solve(W$mat) %*% M)
	ret$mat = M
	ret$p = NULL
  } else if (ortogonal) {
	M = MRT_integerOrtogonal(ret$mat)
	ret$Req = ret$Req %*% (solve(W$mat) %*% M)
	ret$mat = M
	ret$p = NULL
  }
  ret$feq = ret$Req %*% solve(ret$mat)
  ret
}

MRT_feq = function(...) {
	ret = MRT_eq(...)
	ret$feq
}

