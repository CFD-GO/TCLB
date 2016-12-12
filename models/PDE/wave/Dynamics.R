## Model for solving the wave equation as a system of first order DE's
#     u'' = c(u_xx + u_yy)

AddField(name="u")
AddField(name="v")

AddQuantity(name="U")

AddSetting(name="Speed")
AddSetting(name="Value", zonal=TRUE)
AddSetting(name="Viscosity")

AddField(name="u", stencil2d=1)
AddField(name="v", stencil2d=1)

AddNodeType(name="Dirichlet", group="BOUNDARY")
