a2 = 3.852462271644162;
b2 = 0.1304438860971524 * 4.0 ;
c2 = 2.785855170470555;

t = 0.65;

rho2=seq(0,4,len=400)

pow=function(x,y) x^y
p = function(rho2,t) ((rho2*(-pow(b2,3)*pow(rho2,3)/64.+b2*b2*rho2*rho2/16.+b2*rho2/4.+1)*t*c2)/pow(1-b2*rho2/4.,3)-a2*rho2*rho2);

plot(rho2,p,type="l")
