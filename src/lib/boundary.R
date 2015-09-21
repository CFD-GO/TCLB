
FullBounceOp = function(op) {
	cat("real_t tmp;\n")
	tmp=PV("tmp")
        by(Density, Density$group, function(D) {
                i = c("dx","dy","dz")
                D1 = D[,c("name",i)]
                D2 = D1
                D2[,i] = op(D2[,i])
                D3 = merge(D1,D2,by=i)
                D3$name.x < D3$name.y
                D3 = D3[D3$name.x < D3$name.y,]
                for ( i in seq_len(nrow(D3))) {
                        C( tmp, PV(D3$name.x[i]) )
                        C( PV(D3$name.x[i]), PV(D3$name.y[i]) )
                        C( PV(D3$name.y[i]), tmp )
                }
        })
}

FullBounceBack = function() {
	FullBounceOp(function(X) -X);
}

FullSymmetryY = function() {
	FullBounceOp(function(X) {X[2]=-X[2]; X});
}

FullSymmetryZ = function() {
	FullBounceOp(function(X) {X[3]=-X[3]; X});
}


C_pull = function(W, var) {
	ret = div.mod(W[[1]],var)
	cat(var, " = (", ToC(ret[[1]]), ") / (", ToC(ret[[2]]*(-1)), ");\n")
}

ZouHe = function(EQ, direction, sign, type, group="f", P=PV("Pressure"), V=PV("Velocity"), V3) {
	if (missing(V3)) {
		V3 = PV(rep(0,sum(EQ$order == 1)))
		V3[direction] = V
	}
	U = EQ$U

	W1 = cbind(U,i=1:nrow(U))
	W2 = cbind(-U,j=1:nrow(U))
	ret = merge(W1,W2)
	bounce = 1:nrow(U)
	bounce[ret$i] = ret$j

	sel = sign*U[,direction]>0
	fs = f
	Js = c("Jx","Jy","Jz")
	feq = EQ$Req %*% solve(EQ$mat)
	fs[sel] = (feq + (fs-feq)[bounce])[sel]

	presc = PV(rep(0,sum(EQ$order<2)))
	presc[1] = EQ$Req[EQ$order < 2][1]
	presc[direction + 1] = EQ$Req[EQ$order == 2][direction + 1]

	Rs = fs %*% EQ$mat[,EQ$order<2] - presc
	cat("real_t Jx, Jy, Jz, rho;\n")
	rho = PV("rho")
	if (type == "pressure") {
		C( rho, P*3+1);
		C_pull( Rs[1], Js[direction])
	} else if (type == "velocity") {
		nJ = PV("rho") * V
		to_sub = list(nJ[[1]]); names(to_sub) = Js[direction]
		C_pull( subst(Rs[1], to_sub), "rho" ) 
		C( PV(Js[direction]), nJ )
	}
	for (i in 1:(length(Rs)-1)) if (i != direction) C_pull( Rs[i+1], Js[i])
	C(f[sel], fs[sel])
}	

ZouHeNew = function(EQ, f, direction, sign, order, group="f", known="rho",mom) {
  U = EQ$U
  W1 = cbind(U,i=1:nrow(U))
  W2 = cbind(-U,j=1:nrow(U))
  ret = merge(W1,W2)
  bounce = 1:nrow(U)
  bounce[ret$i] = ret$j
  sel = sign*U[,direction]>0
  fs = f
  feq = EQ$feq
  fs[sel] = (feq + (fs-feq)[bounce])[sel]
  Rs = fs %*% EQ$mat
  if (missing(mom)) {
	e = EQ$Req[EQ$order <= order] - Rs[EQ$order <= order]
  } else {
	e = mom - Rs[EQ$order <= order]
  }
  known = c(known, ".M", ToC(f[!sel]))
  all = unique(do.call(c,lapply(fs@vec,function(x) names(x))))
  needed = setdiff(all, known)
  m = do.call(cbind,lapply(e@vec, function(x) apply(x[,needed,drop=FALSE]!=0,2,function(i) sum(abs(x$.M[i])))))
  m = abs(m) > 1e-10
  while (length(needed) > 0) {
    i = which(colSums(m) == 1)[1]
    if (is.na(i)) stop("Cannot solve the boundary problem in ZouHe")
    C_pull(e[i],needed[m[,i]])
    needed = needed[!m[,i]]
    e = e[-i]
    m = m[-m[,i],-i,drop=FALSE]
  }
  C(f[sel],fs[sel])
}
