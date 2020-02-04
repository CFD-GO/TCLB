# Model for solving the wave equation as a system of first order DE's
# u'' = c(u_xx + u_yy)


# Fields are variables (for instance flow-variables, displacements, etc) that are stored in all mesh nodes. 
# Model Dynamics can access these fields with an offset (e.g. field_name(-1,0)).
AddField(name="u", dx=c(-1,1), dy=c(-1,1)) # same as AddField(name="u", stencil2d=1)
AddField(name="v", dx=c(-1,1), dy=c(-1,1))

# 	Outputs:
AddQuantity(name="U")

#	Inputs: Flow Properties
AddSetting(name="Speed")
AddSetting(name="Viscosity")
AddSetting(name="Value", zonal=TRUE)

#	Boundary things:
AddNodeType(name="Dirichlet", group="BOUNDARY")
