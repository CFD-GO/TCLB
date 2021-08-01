
#	declaration of lattice (velocity) directions
x = c(0,1,-1);
P = expand.grid(x=0:2,y=0:2, z=0)
U = expand.grid(x,x,0)

#	declaration of densities
fname = paste("f",P$x,P$y,P$z,sep="")
AddDensity(
	name = fname,
	dx   = U[,1],
	dy   = U[,2],
	comment=paste("LB density fields",1:9-1),
	group="f"
)

AddDensity(name="Init_UX_External", group="init", comment="free stream velocity", parameter=TRUE)
AddDensity(name="Init_UY_External", group="init", comment="free stream velocity", parameter=TRUE)
AddDensity(name="Init_PhaseField_External", group="init", dx=0,dy=0,dz=0, parameter=TRUE)

AddField(name="phaseField_tilde",                 stencil2d=1)

#	Globals - table of global integrals that can be monitored and optimized
AddGlobal(name="PhaseFieldIntegral", comment='Total amount of phasefield', unit="1.")
# 	Outputs:
AddQuantity(name="PhaseField", unit="1.")
AddQuantity(name="Q", unit="1.")


AddStage(name="InitFromFieldsStage", load.densities=TRUE, save.fields=TRUE)
AddAction(name="InitFromFields", "InitFromFieldsStage")

# AddQuantity(name="Random", unit="1.")

#	Boundary things:
AddNodeType(name="DirichletEQ",     group="BOUNDARY")
AddNodeType(name="SRT_DF_SOI",	    group="COLLISION")
AddNodeType(name="SRT_M_SOI",	    group="COLLISION")
AddNodeType(name="TRT_M_SOI",	    group="COLLISION")
AddNodeType(name="TRT_CM_SOI",	    group="COLLISION")
AddNodeType(name="Wall",	        group="BOUNDARY")

# 	Inputs: Flow Properties
AddSetting(name="diffusivity_phi",      default=0.02, comment='Mobility')
AddSetting(name="magic_parameter",      default=1./4., comment='to control relaxation frequency of even moments in TRT collision kernel')
AddSetting(name="lambda", default=1.0, comment="to control intensity of the source term")

AddSetting(name="Init_UX", default=0., comment="free stream x-velocity", zonal=TRUE)
AddSetting(name="Init_UY", default=0., comment="free stream y-velocity", zonal=TRUE)
AddSetting(name="Init_PhaseField",   zonal=TRUE)

#	Benchmark things
AddSetting(name="CylinderCenterX_GH",	default="0", comment='X coord of Gaussian Hill')
AddSetting(name="CylinderCenterY_GH",	default="0", comment='Y coord of Gaussian Hill')
AddSetting(name="Sigma_GH", 		 	default="1", comment='Initial width of the Gaussian Hill', zonal=T)

#	CFD enhancements ;)
AddNodeType(name="Smoothing",               group="ADDITIONALS")  #  To smooth population density during initialization.
AddSetting(name="phase_field_smoothing_coeff", default=0.)     #  To smooth population density during initialization.


# see chapter 10.7.2, eq 10.48, p429 from 'The Lattice Boltzmann Method: Principles and Practice'
# by T. Krüger, H. Kusumaatmaja, A. Kuzmin, O. Shardt, G. Silva, E.M. Viggen
# There are certain values of magic_parameter that show distinctive properties:
# • magic_parameter 1./12 cancels the third-order spatial error, leading to optimal results for pure advection problems.
# • magic_parameter 1./6 cancels the fourth-order spatial error, providing the most accurate results for the pure diffusion equation.
# • magic_parameter 3./16 results in the boundary wall location implemented via bounce-back for the Poiseuille flow exactly in the middle between horizontal walls and fluid nodes.
# • magic_parameter 1./4 provides the most stable simulations.
