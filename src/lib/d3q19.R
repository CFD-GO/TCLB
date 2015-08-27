d3q19_MRTMAT = matrix(c(
         1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
       -30,-11,-11,-11,-11,-11,-11,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,
        12, -4, -4, -4, -4, -4, -4,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
         0,  1, -1,  0,  0,  0,  0,  1, -1,  1, -1,  1, -1,  1, -1,  0,  0,  0,  0,
         0, -4,  4,  0,  0,  0,  0,  1, -1,  1, -1,  1, -1,  1, -1,  0,  0,  0,  0,
         0,  0,  0,  1, -1,  0,  0,  1,  1, -1, -1,  0,  0,  0,  0,  1, -1,  1, -1,
         0,  0,  0, -4,  4,  0,  0,  1,  1, -1, -1,  0,  0,  0,  0,  1, -1,  1, -1,
         0,  0,  0,  0,  0,  1, -1,  0,  0,  0,  0,  1,  1, -1, -1,  1,  1, -1, -1,
         0,  0,  0,  0,  0, -4,  4,  0,  0,  0,  0,  1,  1, -1, -1,  1,  1, -1, -1,
         0,  2,  2, -1, -1, -1, -1,  1,  1,  1,  1,  1,  1,  1,  1, -2, -2, -2, -2,
         0, -4, -4,  2,  2,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1, -2, -2, -2, -2,
         0,  0,  0,  1,  1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1,  0,  0,  0,  0,
         0,  0,  0, -2, -2,  2,  2,  1,  1,  1,  1, -1, -1, -1, -1,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  1, -1, -1,  1,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, -1, -1,  1,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, -1, -1,  1,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  1, -1,  1, -1, -1,  1, -1,  1,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0, -1, -1,  1,  1,  0,  0,  0,  0,  1, -1,  1, -1,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1, -1, -1, -1, -1,  1,  1
    ),19,19)


d3q19_MRT = function (rho="rho", J=c("Jx","Jy","Jz"), R="R", group) {

    MRTMAT = d3q19_MRTMAT

    v = diag(t(MRTMAT) %*% MRTMAT)
    MRTMAT.inv = diag(1/v) %*% t(MRTMAT)

    selU = c(4,6,8)

	if (is.character(rho)) {
	    rho = PV(rho)
	}
	if (is.character(J)) {
		J_str = J
	    J = PV(J)
	}
    u = J * rho^(-1)

    U = MRTMAT[,c(4,6,8)]

    p = ifelse(U < 0,2,U)
    W = NULL; for (i in 1:nrow(p)) {W = cbind(W, apply(t(U) ^ p[i,],2,prod)) }

    H = PV(rep("rho",19));
    for (j in 1:19)
    {
            for (i in 1:3)
            {
                    if (p[j,i] == 1) H[j] = H[j] * J[i] * PV("rho")^(-1)
                    if (p[j,i] == 2) H[j] = H[j] * (J[i]^2 * PV("rho")^(-2) + PV("sigma"))
            }
    }

    feq =  H %*% solve(W)
    Req= subst(feq, sigma=1/3) %*% MRTMAT

	selR = (1:19)[-c(1,selU)]

    Req = gapply(Req, function(x) {i = intersect(names(x),J_str); h=rowSums(x[,i]); sel = h < 3; x[sel,] })


	if (missing(group)) {
		f = NULL;
	} else {
		f = PV(Density$name[Density$group == group])
	}
		
	Rs = PV(rep("rho",19));
	Rs[selU] = J
	if (is.character(R)) {
		if (length(R) == 1) {
			R = PV(R,1:19)
		} else {
			R = PV(R);
		}
	}
	if (length(R) == sum(selR)) {
		Rs[selR] = R
	} else if (length(R) == length(Rs)) {
		Rs[selR] = R[selR]
	} else stop("Bad length of R in d3q19_MRT")

            list(
                    MAT=MRTMAT,
                    Req=Req,
		R=Rs,
		f=f,
		rho=rho,
		J=J,
		U=U,
		selR=selR
            )
}
