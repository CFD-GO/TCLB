# Setting permissive access policy.
#  * This skips checks of fields being overwritten or read prematurely.
#  * Otherwise the model compilation was failing.
#  * This should be removed if the issue is fixed
SetOptions(permissive.access=TRUE)  ### WARNING


if (Options$AllenCahn) {
 	Qname = 'Allen-Cahn'
	DREs <- ('PHI')
 	NumberOfODEs = 0
# 	NumberOfAdditionalParams = 1
	Params  <- c("Lambda")

} else if (Options$SIR_SimpleLaplace) {
    Qname = 'SIR_SimpleLaplace'
	#NumberOfDREs = 3
	DREs  <- c("S", "I", "R")
	NumberOfODEs = 0
	Params  <- c("Beta", "Gamma")

} else if (Options$SIR_ModifiedPeng) {
   	Qname = 'SIR_ModifiedPeng'
	# NumberOfDREs = 1
	# NumberOfODEs = 3+1
	# NumberOfAdditionalParams = 4

	DREs  <- c("W")
	ODEs  <- c("S", "I", "R", "N")
	Params  <- c("Beta", "Beta_w", "Gamma")


} else if (Options$SimpleDiffusion) {
   	Qname = 'SimpleDiffusion'
	DREs <- ('PHI')

	NumberOfAdditionalParams = 0	
 	NumberOfODEs = 0

} else if (Options$LinearReaction) {
   	Qname = 'LinearReaction'
	DREs <- ('PHI')

	Params  <- c("LinearReactionRate")
 	NumberOfODEs = 0

}

##END MANUAL CONFIG
if (Options$Heun) {
	QIntegrator = 'Heun'
} else if (Options$Euler) {
	QIntegrator = 'Euler'
} else if (Options$Trapezoid) {
	QIntegrator = 'Trapezoid'
} else if (Options$Midpoint) {
	QIntegrator = 'Midpoint'
} else {
	QIntegrator = 'Trapezoid'
}

if (exists('DREs'))  {
	NumberOfDREs = length(DREs)
} 
	
if (!exists('DREs') && NumberOfDREs > 0 )  {
	DREs = paste('DRE', seq(1, NumberOfDREs, 1), sep="_" )
}

if (exists('ODEs') ) {
	NumberOfODEs = length(ODEs)
} 
	
if (!exists('ODEs') && NumberOfODEs > 0 )  {
	ODEs = paste('ODE', seq(1, NumberOfODEs, 1), sep="_" )
}

if (exists('Params')) {
	NumberOfAdditionalParams = length(Params)
} 
	
if (!exists('Params') && NumberOfAdditionalParams > 0 )  {
	Params = paste('C', seq(1, NumberOfAdditionalParams, 1), sep="_" )
}


if (NumberOfDREs > 0){
	dre_loop = function(expr) {
		for (i in seq(1, NumberOfDREs, 1)) {
			expr(i)
		}		
	}
} else {
	dre_loop = function(expr) {}		
}

if (NumberOfODEs > 0){
	ode_loop = function(expr) {
		for (i in seq(1, NumberOfODEs, 1)) {
			expr(i)
		}		
	}
} else {
	ode_loop = function(expr) {}		
}

# declaration of lattice (velocity) directions
x = c(0,1,-1);
P = expand.grid(x=0:2,y=0:2, z=0)
U = expand.grid(x,x,0)

dre_loop(function(i){
# declaration of densities
	gname = paste("dre",i,sep="_")
	bname = paste(gname, 'f',sep="_")
	fname = paste(bname,P$x,P$y,P$z,sep="")

	AddDensity(
		name = fname,
		dx   = U[,1],
		dy   = U[,2],
		comment=paste("LB density fields ",gname),
		group=gname
	)
})



ode_loop(function(i) {
	bname = paste('ode', i, sep="_")
	AddField(name=bname, dx=c(-1,1), dy=c(-1,1)) # same as AddField(name="phi", stencil2d=1)

	bname = ODEs[i]
	exname = paste('Init', bname, 'External', sep="_")
	AddDensity(name=exname, group="init", dx=0,dy=0,dz=0, parameter=TRUE)
})

# 	Outputs:

dre_loop(function(i){
	bname = DREs[i]
	AddQuantity(name=bname, unit="1.")
	
	exname = paste('Init', bname, 'External', sep="_")
	AddDensity(name=exname, group="init", dx=0,dy=0,dz=0, parameter=TRUE)
})


ode_loop(function(i) {
	bname = ODEs[i]
	AddQuantity(name=bname, unit="1.")

	bname = paste('Init', ODEs[i] , sep="_")
	AddSetting(name=bname, zonal=TRUE)	
})

AddStage(name="InitFromExternal", load.densities=TRUE, save.fields=TRUE)
AddAction(name="InitFromExternalAction", "InitFromExternal")


#	Boundary things:
AddNodeType(name="Wall",	    group="BOUNDARY")

AddNodeType(name="SRT_DF",	    group="COLLISION")
AddNodeType(name="TRT_M",	    group="COLLISION")

# Inputs: Flow Properties
AddSetting(name="magic_parameter",      default=1./6., comment='to control relaxation frequency of even moments in TRT collision kernel')

dre_loop(function(i) {
	bname = paste('Init', DREs[i], sep="_")
	AddSetting(name=bname, zonal=TRUE)

	bname = paste('Diffusivity',  DREs[i] , sep="_")
	comment = paste('Diffusivity for ', DREs[i], sep="_")
	AddSetting(name=bname,	default=0.02, comment=comment)
})

if (NumberOfAdditionalParams > 0) {
	for (i in seq(1, NumberOfAdditionalParams)){
		bname = Params[i]
		comment = paste('Model parameter ', Params[i])
		AddSetting(name=bname, default=0.0, comment=comment)
	}
}

Extra_Dynamics_C_Header = "
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#include <Eigen/Dense>
"

# see chapter 10.7.2, eq 10.48, p429 from 'The Lattice Boltzmann Method: Principles and Practice'
# by T. Krüger, H. Kusumaatmaja, A. Kuzmin, O. Shardt, G. Silva, E.M. Viggen
# There are certain values of magic_parameter that show distinctive properties:
# • magic_parameter 1./12 cancels the third-order spatial error, leading to optimal results for pure advection problems.
# • magic_parameter 1./6 cancels the fourth-order spatial error, providing the most accurate results for the pure diffusion equation.
# • magic_parameter 3./16 results in the boundary wall location implemented via bounce-back for the Poiseuille flow exactly in the middle between horizontal walls and fluid nodes.
# • magic_parameter 1./4 provides the most stable simulations.
