mark.stensil = function(tab,mx=0,only2d=FALSE,pref="st_") {
        if (! "minx" %in% names(tab)) {
                tab$minx = tab$maxx = tab$dx
                tab$miny = tab$maxy = tab$dy
                tab$minz = tab$maxz = tab$dz
        }
        mx = max(abs(c(mx, tab$minx, tab$miny, tab$minz, tab$maxx, tab$maxy, tab$maxz)))

        e = 0.9
        a = 0.5 * pi/2
        b = 0.3 * pi/2
        if (! only2d) {
                mat = t(matrix(c(cos(a),sin(a)*sin(b),0,cos(b),sin(a),-cos(a)*sin(b)),2,3))
          ret = cbind(c(mx+1,-mx), 0, c(0,1)) %*% mat
          ddx = diff(ret[,1])
          if (ddx < 0.01) {
                  mat[3,1] = mat[3,1] - ddx + 0.01
          }
        } else {
                mat = t(matrix(c(1,0,0,1,0,0),2,3))
        }

        p = cbind(
                x=c(-0.5,0.5,0.5,-0.5),
                y=c(-0.5,-0.5,0.5,0.5),
                z=0
        )*e
        po3 = function(x,y,z,...) {
                np = cbind(p[,1]+x,p[,2]+y,p[,3]+z)
                polygon(np %*% mat,...)
        }

        w = c(tab$minx, tab$miny, tab$minz, tab$maxx, tab$maxy, tab$maxz)
        w = paste(ifelse(w<0,"n","p"),abs(w),sep="",collapse="")
        w = paste(ifelse(only2d,"a","b"),mx,w,sep="")
        w = paste(pref,w,".png",sep="")

	sz = 50
        png(w,width=sz*2,height=sz,bg = "transparent",res=200)

        par(mar=c(0,0,0,0))
        maxy = c(mx+0.5,mx+0.5,-mx) %*% mat[,2]
        maxx = c(mx+0.5,0,mx) %*% mat[,1]
        if (maxx > maxy*2) maxy = maxx/2
        plot(NA, xlim=c(-maxy,maxy)*2, ylim=c(-maxy,maxy), bty='n', xaxt='n', yaxt='n')

        ntab = expand.grid(x=-mx:mx,y=-mx:mx,z=-mx:mx)
        for (i in 1:nrow(ntab)) {
                d = ntab[i,]
                if (
                        (d$x >= tab$minx) &
                        (d$y >= tab$miny) &
                        (d$z >= tab$minz) &
                        (d$x <= tab$maxx) &
                        (d$y <= tab$maxy) &
                        (d$z <= tab$maxz) ) {
                        po3(d$x,d$y,d$z,col=3)
                } else {
                        po3(d$x,d$y,d$z)
                }
        }
        dev.off()
	paste0("/",w)
}


