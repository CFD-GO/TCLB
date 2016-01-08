library(polyAlgebra)

all.subsets = function(n) {
  as.matrix(do.call(expand.grid,rep(list(c(TRUE,FALSE)),n)))
}

FCT_eq = function(I,M,K) {
v = c("X","Y","Z")
nn = 3

ord = function(p) {
  ret = colSums(outer(p,v,"=="))
  sum(ret * nn^(0:2))+1
}

opt.rec = function(p,M) {
  w = all.subsets(length(p))
  ret = lapply(1:length(p), function(s) {
    ret = PV(0)
    for (j in which(w[,s]==FALSE)) {
      i = w[j,]
      ret = ret + M[ord(p[i])] * K[ord(p[!i])]
    }
    ret
  })
  ret_eq = sapply(ret,ToC)
  ret_eq = sapply(strsplit(ret_eq,split="*",fixed = TRUE),length)
  i = which.min(ret_eq)
  ret[[i]]
}

to.p = function(w) {
  do.call(c,mapply(rep,v[1:length(w)],w,SIMPLIFY = FALSE))
}

nM = M
for (i in 2:nrow(I)) {
  p = to.p(I[i,])
  nM[ord(p)] = opt.rec(p,M)
}
nM
}


