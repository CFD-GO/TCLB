
Bounce = function(U, FUN=function(U) {-U} ) {
	W1 = cbind(U,i=1:nrow(U))
	W2 = cbind(-U,j=1:nrow(U))
	ret = merge(W1,W2)
	bounce = 1:nrow(U)
	bounce[ret$i] = ret$j
	bounce
}


FullBounceOp = function(op, group) {
	cat("real_t tmp;\n")
	tmp=PV("tmp")
        by(Density, Density$group, function(D) {
                 if (  group == '' || D[1,'group'] %in% group ) {  
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
                }
        })
}

FullBounceBack = function(group='') {
	FullBounceOp(function(X) -X, group);
}

Symmetry  = function(direction,sign,group='') {
        by(Density, Density$group, function(D) {
                if (  group == '' || D[1,'group'] %in% group ) {
                    i = c("dx","dy","dz")
                    D1 = D[,c("name",i)]
                    D2 = D1
                    D2[,direction + 1] = -D2[,direction + 1]
                    D3 = merge(D1,D2,by=i)
                    D3$name.x < D3$name.y
                    D3 = D3[D3$name.x < D3$name.y,]
                    for ( i in seq_len(nrow(D3))) {
                            if (sign*D3[,direction][i] > 0) C( PV(D3$name.x[i]), PV(D3$name.y[i]) )
                            else C( PV(D3$name.y[i]), PV(D3$name.x[i]) )
                    }
                }
        })
}

C_pull = function(W, var) {
	ret = div.mod(W[[1]],var)
	A = ret[[1]]*(-1)
	B = ret[[2]]

	if (nrow(B) > 1) {
		cat(var, " = (", ToC(A), ") / (", ToC(B), ");\n")
	} else {
		cat(var, " = ", ToC(A * (B ** -1)), ";\n")
	}
}

ZouHe = function(EQ, direction, sign, type, group=f, P=PV("Pressure"), V=PV("Velocity"), V3, predefined="false") {
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
	fs = group
	Js = c("Jx","Jy","Jz")
	feq = EQ$Req %*% solve(EQ$mat)
	fs[sel] = (feq + (fs-feq)[bounce])[sel]

	presc = PV(rep(0,sum(EQ$order<2)))
	presc[1] = EQ$Req[EQ$order < 2][1]
	presc[direction + 1] = EQ$Req[EQ$order == 2][direction + 1]

	Rs = fs %*% EQ$mat[,EQ$order<2] - presc
	if (predefined == "false")
	{
		cat("real_t Jx, Jy, Jz, rho;\n")
	}	
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
	for (i in 1:(length(Rs)-1)) if ((i != direction) && !(is.zero(V3[i]))) C(PV(Js[i]),PV(Js[i])+ rho*V3[i])
	C(group[sel], fs[sel])
}	

ZouHeNew = function(EQ, f, direction, sign, order, group=f, known="rho",mom) {
  U = EQ$U
  W1 = cbind(U,i=1:nrow(U))
  W2 = cbind(-U,j=1:nrow(U))
  ret = merge(W1,W2)
  bounce = 1:nrow(U)
  bounce[ret$i] = ret$j
  sel = sign*U[,direction]>0
  fs = group
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

ZouHeRewrite = function(EQ, f, n, type=c("velocity","pressure","do nothing"), rhs) {
  # --- Prepare arguments
	type=match.arg(type)

    d = length(n)
	if (sum(n == 0) != d-1) stop("Normal have to be cartesian")
	if (sum(abs(n)) != 1)   stop("Normal have to be cartesian versor")

	if (missing(rhs)) rhs = switch(type,
		velocity=PV("Velocity")*abs(n),
		pressure=PV("Pressure")*3+1,
		'do nothing'=PV("Pressure")+1./3.
	)

	direction = which(n != 0)
	if (is.data.frame(EQ) || is.matrix(EQ)) {
		U = as.matrix(EQ)
	} else {
		U = EQ$U
	}
  # --- Create a new F equilibrum with 'R' as momentum
	R = paste0("R",c("x","y","z"))[1:d]
	EQ2 = MRT_eq(U, ortogonal=FALSE, J=PV(R))
	feq = EQ2$feq
	sel = as.vector((U %*% n) < 0)
  # --- Creating new 'fs' that has symetric non-equilibrum part
	bounce = Bounce(U)
	fs = f; fs[sel] = (feq + (fs - feq)[bounce])[sel]
  # --- Preparing moments to set
	if (type == "do nothing") {
	  # --- Set 2nd order moment tensor times normal vector
        stop("This will not work, use pressure/velocity ZouHe")
		#rhs = rhs * n
		#qn = fs %*% EQ2$D2 %*% n
	} else if (type == "pressure") {
          # --- Set density and non-normal velocity compoments
		eqn = V(fs %*% EQ2$U %*% diag(d))
		eqn[direction] = V(sum(fs))
		rhs = rhs * abs(n);
	} else if (type == "velocity") {
          # --- Set all velocity components
		eqn = V( fs %*% EQ2$U %*% diag(d))
		rhs = rhs * sum(fs)
	} else stop("Unknown type in ZouHe")
	cat("/********* ", type, "-type Zou He boundary condition  ****************/\n",sep="");
	eqn = eqn - rhs;
	if (length(eqn) != length(R)) stop("Something is terribly wrong")
  # --- Solving all equations for 'R'
    i0 = (1:length(R))[ abs(n) != 0  ]
    cat("real_t "); C_pull(eqn[i0],R[i0]);
    sl = 1:length(R)
    sl = sl[sl != i0]
	for (i in sl) { cat("real_t "); C_pull(eqn[i],R[i]); }
  # --- Setting the missing densities f
	C(f[sel],fs[sel]);
}

