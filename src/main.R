t1 = 4/9
t2 = 1/9
t3 = 1/36
c_squ = 1/3
U = matrix(c(
 1, 0,
 1, 1,
 0, 1,
-1, 1,
-1, 0,
-1,-1,
 0,-1,
 1,-1,
 0, 0
),2,9)
U = t(U)





makeplot = function (...) {
	I = matrix(d,nx,ny)
	I[B!=0] = NA
	col = colorRamp(c("yellow","red"))
	col = col(seq(0,1,len=100))
	col = rgb(col,max=255)
	image(1:nx,1:ny,I,col=col,...)
#	image(1:nx,1:ny,matrix(bound,nx,ny))
	scale=1/sqrt(max(rowSums(u**2)))*2
	segments(p[,1],p[,2],p[,1]+scale*u[,1],p[,2]+scale*u[,2])
}

tn = diag(c(t2,t3,t2,t3,t2,t3,t2,t3,t1))

for (i in 1:100)
{
 filename = sprintf("out%05d.txt", i);
 if (!file.exists(filename)) filename="out.txt"
 tab = as.matrix(read.table(filename))

nx = nrow(tab)
ny = ncol(tab)/10
f = matrix(t(tab),nrow=10)
f = t(f)
B = f[,10]
f = f[,-10]
p = cbind(rep(1:nx,times=ny),rep(1:ny,each=nx))
u = f %*% U
d = rowSums(f)
makeplot()
print(range(f))
print(range(u))
print(range(1-d))
 if (filename=="out.txt") break;
}