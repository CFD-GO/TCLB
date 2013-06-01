
# MODELS: 
#	d1q3, d2q9, d2q9_adj, d2q9_adj_smooth, d2q9_adj_top, d2q9_cor, d2q9_entropic, d2q9_exp, d2q9_heat, d2q9_heat_adj, d2q9_heat_diff, d2q9_kuper, d2q9_kuper2, d2q9_mes, d2q9_rho, d2q9_transport, d2q9_transport_adj, d2q9_two, d3q19, d3q19_adj, d3q19_heat, d3q19_heat_adj, d3q19_n

include conf.mk
ADJOINT=0        # calculate adjoint: 1-on, 0-off
GRAPHICS=1       # GLUT graphics: 1-on, 0-off
GRID3D=0         # use 3D block grid (only avaliable on capability 2.x): 1-on, 0-off
ARCH=sm_11       # CUDA architecture: sm_10 for capability 1.0, sm_13 for capability 1.3 etc.
DOUBLE=1         # precision: 1-double, 0-float


#######################################################################################################################

all: d1q3 d2q9 d2q9_adj d2q9_adj_smooth d2q9_adj_top d2q9_cor d2q9_entropic d2q9_exp d2q9_heat d2q9_heat_adj d2q9_heat_diff d2q9_kuper d2q9_kuper2 d2q9_mes d2q9_rho d2q9_transport d2q9_transport_adj d2q9_two d3q19 d3q19_adj d3q19_heat d3q19_heat_adj d3q19_n

ifeq '$(strip $(STANDALONE))' '1'
 total : sa
else
total : Rpackage
	R CMD INSTALL CLB_0.00.tar.gz
endif

makefile:src/makefile.main.Rt
	tools/RT -f $< -o $@

thor : Rpackage
	scp CLB_0.00.tar.gz tachion:cuwork

Rpackage : source package/configure
	R CMD build package
	
package/configure:package/configure.ac
	@echo "AUTOCONF     $@"
	@cd package; autoconf

sa : source
	@cd standalone; $(MAKE)

MPI_INCLUDES = /usr/include/mpi/
MPI_LIBS     = /usr/lib/mpich/lib/
MPI_OPT      = -L$(MPI_LIBS) -I$(MPI_INCLUDES) -lmpi #-Xptxas -v
RT = tools/RT
ADMOD = tools/ADmod.R
MAKEHEADERS = tools/makeheaders

SRC=src

#ifeq '$(strip $(STANDALONE))' '1'
 DEST=standalone
 ADDITIONALS=makefile model README.md
 SOURCE_CU+=main.cu
 HEADERS_H+=DataLine.h
#else
# DEST=package/src
# ADDITIONALS=package/src/Makefile.in package/data/LB.RData
#endif

SOURCE_CU+=Global.cu Lattice.cu vtkLattice.cu vtkOutput.cu cross.cu cuda.cu LatticeContainer.cu Dynamics.c inter.cpp Solver.cpp pugixml.cpp Geometry.cu def.cpp
SOURCE_R=conf.R Dynamics.R
SOURCE=$(addprefix $(DEST)/,$(SOURCE_CU))
HEADERS_H+=Global.h gpu_anim.h LatticeContainer.h Lattice.h Node.h Region.h vtkLattice.h vtkOutput.h cross.h gl_helper.h Dynamics.h Dynamics.hp types.h Node_types.h Solver.h pugixml.hpp pugiconfig.hpp Geometry.h def.h utils.h
HEADERS=$(addprefix $(DEST)/,$(HEADERS_H))

ALL_FILES=$(SOURCE_CU) $(HEADERS_H) $(ADDITIONALS)
DEST_FILES=$(addprefix $(DEST)/,$(ALL_FILES))

AOUT=main

CC=nvcc
CCTXT=NVCC

RTOPT=

OPT=$(MPI_OPT)

ifdef MODEL
 RTOPT+=MODEL=\"$(strip $(MODEL))\"
endif

ifdef ADJOINT
 RTOPT+=ADJOINT=$(strip $(ADJOINT))
endif

ifdef DOUBLE
 RTOPT+=DOUBLE=$(strip $(DOUBLE))
endif

ifdef GRAPHICS
 RTOPT+=GRAPHICS=$(strip $(GRAPHICS))
endif

ifeq '$(strip $(ADJOINT))' '1'
 OPT+=-D ADJOINT
 SOURCE_CU+=Dynamics_b.c ADTools.cu Dynamics_adj.c
 HEADERS_H+=Dynamics_b.hp Dynamics_b.h types_b.h ADpre_.h Dynamics_adj.hp ADpre__b.h
endif

ifeq '$(strip $(GRAPHICS))' '1'
 OPT+=-D GRAPHICS -lglut
endif

ifeq '$(strip $(DIRECT_MEM))' '1'
 OPT+=-D DIRECT_MEM
endif

ifdef ARCH
 OPT+=-arch $(strip $(ARCH))
endif

ifeq '$(strip $(GRID3D))' '1'
 OPT+=-D GRID_3D
endif

ifeq '$(strip $(DOUBLE))' '1'
 OPT+=-D CALC_DOUBLE_PRECISION
endif

MODELPATH=$(strip $(MODEL))

.PRECIOUS:$(DEST_FILES)

source:Dynamics.R conf.R $(DEST_FILES)
	@cd $(DEST); git add $(ALL_FILES)


package/data/LB.RData: conf.R Dynamics.R
	@echo "MAKEDATA     $@"
	@Rscript tools/makeData.R

clear:
	@echo "  RM         ALL"
	@rm `ls | grep -v -e ^makefile$$ -e .mk$$` 2>/dev/null; true

$(DEST)/Dynamics_b.c $(DEST)/Dynamics_b.h $(DEST)/types_b.h $(DEST)/ADpre_.h $(DEST)/ADpre__b.h : tapenade.run

.INTERMEDIATE : tapenade.run

tapenade.run : $(DEST)/Dynamics.c $(DEST)/ADpre.h $(DEST)/ADpre_b.h $(DEST)/Dynamics.h $(DEST)/Node_types.h $(DEST)/types.h $(DEST)/ADset.sh
	@echo "  TAPENADE   $<"
	@(cd standalone; ../tools/makeAD)

###############################################################################
######       AUTO GENERATED CASES for RT                                 ######
###############################################################################


d1q3:  | standalone/d1q3

d1q3: standalone/d1q3 $(addprefix standalone/d1q3/,$(ALL_FILES))
	@echo "  DONE       $@"

standalone/d1q3 :
	mkdir -p $@

d2q9:  | standalone/d2q9

d2q9: standalone/d2q9 $(addprefix standalone/d2q9/,$(ALL_FILES))
	@echo "  DONE       $@"

standalone/d2q9 :
	mkdir -p $@

d2q9_adj:  | standalone/d2q9_adj

d2q9_adj: standalone/d2q9_adj $(addprefix standalone/d2q9_adj/,$(ALL_FILES))
	@echo "  DONE       $@"

standalone/d2q9_adj :
	mkdir -p $@

d2q9_adj_smooth:  | standalone/d2q9_adj_smooth

d2q9_adj_smooth: standalone/d2q9_adj_smooth $(addprefix standalone/d2q9_adj_smooth/,$(ALL_FILES))
	@echo "  DONE       $@"

standalone/d2q9_adj_smooth :
	mkdir -p $@

d2q9_adj_top:  | standalone/d2q9_adj_top

d2q9_adj_top: standalone/d2q9_adj_top $(addprefix standalone/d2q9_adj_top/,$(ALL_FILES))
	@echo "  DONE       $@"

standalone/d2q9_adj_top :
	mkdir -p $@

d2q9_cor:  | standalone/d2q9_cor

d2q9_cor: standalone/d2q9_cor $(addprefix standalone/d2q9_cor/,$(ALL_FILES))
	@echo "  DONE       $@"

standalone/d2q9_cor :
	mkdir -p $@

d2q9_entropic:  | standalone/d2q9_entropic

d2q9_entropic: standalone/d2q9_entropic $(addprefix standalone/d2q9_entropic/,$(ALL_FILES))
	@echo "  DONE       $@"

standalone/d2q9_entropic :
	mkdir -p $@

d2q9_exp:  | standalone/d2q9_exp

d2q9_exp: standalone/d2q9_exp $(addprefix standalone/d2q9_exp/,$(ALL_FILES))
	@echo "  DONE       $@"

standalone/d2q9_exp :
	mkdir -p $@

d2q9_heat:  | standalone/d2q9_heat

d2q9_heat: standalone/d2q9_heat $(addprefix standalone/d2q9_heat/,$(ALL_FILES))
	@echo "  DONE       $@"

standalone/d2q9_heat :
	mkdir -p $@

d2q9_heat_adj:  | standalone/d2q9_heat_adj

d2q9_heat_adj: standalone/d2q9_heat_adj $(addprefix standalone/d2q9_heat_adj/,$(ALL_FILES))
	@echo "  DONE       $@"

standalone/d2q9_heat_adj :
	mkdir -p $@

d2q9_heat_diff:  | standalone/d2q9_heat_diff

d2q9_heat_diff: standalone/d2q9_heat_diff $(addprefix standalone/d2q9_heat_diff/,$(ALL_FILES))
	@echo "  DONE       $@"

standalone/d2q9_heat_diff :
	mkdir -p $@

d2q9_kuper:  | standalone/d2q9_kuper

d2q9_kuper: standalone/d2q9_kuper $(addprefix standalone/d2q9_kuper/,$(ALL_FILES))
	@echo "  DONE       $@"

standalone/d2q9_kuper :
	mkdir -p $@

d2q9_kuper2:  | standalone/d2q9_kuper2

d2q9_kuper2: standalone/d2q9_kuper2 $(addprefix standalone/d2q9_kuper2/,$(ALL_FILES))
	@echo "  DONE       $@"

standalone/d2q9_kuper2 :
	mkdir -p $@

d2q9_mes:  | standalone/d2q9_mes

d2q9_mes: standalone/d2q9_mes $(addprefix standalone/d2q9_mes/,$(ALL_FILES))
	@echo "  DONE       $@"

standalone/d2q9_mes :
	mkdir -p $@

d2q9_rho:  | standalone/d2q9_rho

d2q9_rho: standalone/d2q9_rho $(addprefix standalone/d2q9_rho/,$(ALL_FILES))
	@echo "  DONE       $@"

standalone/d2q9_rho :
	mkdir -p $@

d2q9_transport:  | standalone/d2q9_transport

d2q9_transport: standalone/d2q9_transport $(addprefix standalone/d2q9_transport/,$(ALL_FILES))
	@echo "  DONE       $@"

standalone/d2q9_transport :
	mkdir -p $@

d2q9_transport_adj:  | standalone/d2q9_transport_adj

d2q9_transport_adj: standalone/d2q9_transport_adj $(addprefix standalone/d2q9_transport_adj/,$(ALL_FILES))
	@echo "  DONE       $@"

standalone/d2q9_transport_adj :
	mkdir -p $@

d2q9_two:  | standalone/d2q9_two

d2q9_two: standalone/d2q9_two $(addprefix standalone/d2q9_two/,$(ALL_FILES))
	@echo "  DONE       $@"

standalone/d2q9_two :
	mkdir -p $@

d3q19:  | standalone/d3q19

d3q19: standalone/d3q19 $(addprefix standalone/d3q19/,$(ALL_FILES))
	@echo "  DONE       $@"

standalone/d3q19 :
	mkdir -p $@

d3q19_adj:  | standalone/d3q19_adj

d3q19_adj: standalone/d3q19_adj $(addprefix standalone/d3q19_adj/,$(ALL_FILES))
	@echo "  DONE       $@"

standalone/d3q19_adj :
	mkdir -p $@

d3q19_heat:  | standalone/d3q19_heat

d3q19_heat: standalone/d3q19_heat $(addprefix standalone/d3q19_heat/,$(ALL_FILES))
	@echo "  DONE       $@"

standalone/d3q19_heat :
	mkdir -p $@

d3q19_heat_adj:  | standalone/d3q19_heat_adj

d3q19_heat_adj: standalone/d3q19_heat_adj $(addprefix standalone/d3q19_heat_adj/,$(ALL_FILES))
	@echo "  DONE       $@"

standalone/d3q19_heat_adj :
	mkdir -p $@

d3q19_n:  | standalone/d3q19_n

d3q19_n: standalone/d3q19_n $(addprefix standalone/d3q19_n/,$(ALL_FILES))
	@echo "  DONE       $@"

standalone/d3q19_n :
	mkdir -p $@




# for model d1q3 and destination standalone

standalone/d1q3/%:$(SRC)/%.Rt $(SRC)/d1q3/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d1q3 -o $@ $(RTOPT) || rm $@

standalone/d1q3/%:$(SRC)/d1q3/%.Rt $(SRC)/d1q3/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d1q3 -o $@ $(RTOPT) || rm $@

standalone/d1q3/%:$(SRC)/d1q3/%
	@echo "  CP         $@ (model)"
	@cp $< $@

standalone/d1q3/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d1q3 and destination package/src

package/src/d1q3/%:$(SRC)/%.Rt $(SRC)/d1q3/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d1q3 -o $@ $(RTOPT) || rm $@

package/src/d1q3/%:$(SRC)/d1q3/%.Rt $(SRC)/d1q3/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d1q3 -o $@ $(RTOPT) || rm $@

package/src/d1q3/%:$(SRC)/d1q3/%
	@echo "  CP         $@ (model)"
	@cp $< $@

package/src/d1q3/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d2q9 and destination standalone

standalone/d2q9/%:$(SRC)/%.Rt $(SRC)/d2q9/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9 -o $@ $(RTOPT) || rm $@

standalone/d2q9/%:$(SRC)/d2q9/%.Rt $(SRC)/d2q9/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9 -o $@ $(RTOPT) || rm $@

standalone/d2q9/%:$(SRC)/d2q9/%
	@echo "  CP         $@ (model)"
	@cp $< $@

standalone/d2q9/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d2q9 and destination package/src

package/src/d2q9/%:$(SRC)/%.Rt $(SRC)/d2q9/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9 -o $@ $(RTOPT) || rm $@

package/src/d2q9/%:$(SRC)/d2q9/%.Rt $(SRC)/d2q9/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9 -o $@ $(RTOPT) || rm $@

package/src/d2q9/%:$(SRC)/d2q9/%
	@echo "  CP         $@ (model)"
	@cp $< $@

package/src/d2q9/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d2q9_adj and destination standalone

standalone/d2q9_adj/%:$(SRC)/%.Rt $(SRC)/d2q9_adj/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_adj -o $@ $(RTOPT) || rm $@

standalone/d2q9_adj/%:$(SRC)/d2q9_adj/%.Rt $(SRC)/d2q9_adj/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_adj -o $@ $(RTOPT) || rm $@

standalone/d2q9_adj/%:$(SRC)/d2q9_adj/%
	@echo "  CP         $@ (model)"
	@cp $< $@

standalone/d2q9_adj/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d2q9_adj and destination package/src

package/src/d2q9_adj/%:$(SRC)/%.Rt $(SRC)/d2q9_adj/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_adj -o $@ $(RTOPT) || rm $@

package/src/d2q9_adj/%:$(SRC)/d2q9_adj/%.Rt $(SRC)/d2q9_adj/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_adj -o $@ $(RTOPT) || rm $@

package/src/d2q9_adj/%:$(SRC)/d2q9_adj/%
	@echo "  CP         $@ (model)"
	@cp $< $@

package/src/d2q9_adj/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d2q9_adj_smooth and destination standalone

standalone/d2q9_adj_smooth/%:$(SRC)/%.Rt $(SRC)/d2q9_adj_smooth/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_adj_smooth -o $@ $(RTOPT) || rm $@

standalone/d2q9_adj_smooth/%:$(SRC)/d2q9_adj_smooth/%.Rt $(SRC)/d2q9_adj_smooth/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_adj_smooth -o $@ $(RTOPT) || rm $@

standalone/d2q9_adj_smooth/%:$(SRC)/d2q9_adj_smooth/%
	@echo "  CP         $@ (model)"
	@cp $< $@

standalone/d2q9_adj_smooth/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d2q9_adj_smooth and destination package/src

package/src/d2q9_adj_smooth/%:$(SRC)/%.Rt $(SRC)/d2q9_adj_smooth/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_adj_smooth -o $@ $(RTOPT) || rm $@

package/src/d2q9_adj_smooth/%:$(SRC)/d2q9_adj_smooth/%.Rt $(SRC)/d2q9_adj_smooth/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_adj_smooth -o $@ $(RTOPT) || rm $@

package/src/d2q9_adj_smooth/%:$(SRC)/d2q9_adj_smooth/%
	@echo "  CP         $@ (model)"
	@cp $< $@

package/src/d2q9_adj_smooth/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d2q9_adj_top and destination standalone

standalone/d2q9_adj_top/%:$(SRC)/%.Rt $(SRC)/d2q9_adj_top/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_adj_top -o $@ $(RTOPT) || rm $@

standalone/d2q9_adj_top/%:$(SRC)/d2q9_adj_top/%.Rt $(SRC)/d2q9_adj_top/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_adj_top -o $@ $(RTOPT) || rm $@

standalone/d2q9_adj_top/%:$(SRC)/d2q9_adj_top/%
	@echo "  CP         $@ (model)"
	@cp $< $@

standalone/d2q9_adj_top/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d2q9_adj_top and destination package/src

package/src/d2q9_adj_top/%:$(SRC)/%.Rt $(SRC)/d2q9_adj_top/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_adj_top -o $@ $(RTOPT) || rm $@

package/src/d2q9_adj_top/%:$(SRC)/d2q9_adj_top/%.Rt $(SRC)/d2q9_adj_top/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_adj_top -o $@ $(RTOPT) || rm $@

package/src/d2q9_adj_top/%:$(SRC)/d2q9_adj_top/%
	@echo "  CP         $@ (model)"
	@cp $< $@

package/src/d2q9_adj_top/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d2q9_cor and destination standalone

standalone/d2q9_cor/%:$(SRC)/%.Rt $(SRC)/d2q9_cor/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_cor -o $@ $(RTOPT) || rm $@

standalone/d2q9_cor/%:$(SRC)/d2q9_cor/%.Rt $(SRC)/d2q9_cor/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_cor -o $@ $(RTOPT) || rm $@

standalone/d2q9_cor/%:$(SRC)/d2q9_cor/%
	@echo "  CP         $@ (model)"
	@cp $< $@

standalone/d2q9_cor/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d2q9_cor and destination package/src

package/src/d2q9_cor/%:$(SRC)/%.Rt $(SRC)/d2q9_cor/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_cor -o $@ $(RTOPT) || rm $@

package/src/d2q9_cor/%:$(SRC)/d2q9_cor/%.Rt $(SRC)/d2q9_cor/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_cor -o $@ $(RTOPT) || rm $@

package/src/d2q9_cor/%:$(SRC)/d2q9_cor/%
	@echo "  CP         $@ (model)"
	@cp $< $@

package/src/d2q9_cor/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d2q9_entropic and destination standalone

standalone/d2q9_entropic/%:$(SRC)/%.Rt $(SRC)/d2q9_entropic/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_entropic -o $@ $(RTOPT) || rm $@

standalone/d2q9_entropic/%:$(SRC)/d2q9_entropic/%.Rt $(SRC)/d2q9_entropic/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_entropic -o $@ $(RTOPT) || rm $@

standalone/d2q9_entropic/%:$(SRC)/d2q9_entropic/%
	@echo "  CP         $@ (model)"
	@cp $< $@

standalone/d2q9_entropic/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d2q9_entropic and destination package/src

package/src/d2q9_entropic/%:$(SRC)/%.Rt $(SRC)/d2q9_entropic/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_entropic -o $@ $(RTOPT) || rm $@

package/src/d2q9_entropic/%:$(SRC)/d2q9_entropic/%.Rt $(SRC)/d2q9_entropic/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_entropic -o $@ $(RTOPT) || rm $@

package/src/d2q9_entropic/%:$(SRC)/d2q9_entropic/%
	@echo "  CP         $@ (model)"
	@cp $< $@

package/src/d2q9_entropic/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d2q9_exp and destination standalone

standalone/d2q9_exp/%:$(SRC)/%.Rt $(SRC)/d2q9_exp/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_exp -o $@ $(RTOPT) || rm $@

standalone/d2q9_exp/%:$(SRC)/d2q9_exp/%.Rt $(SRC)/d2q9_exp/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_exp -o $@ $(RTOPT) || rm $@

standalone/d2q9_exp/%:$(SRC)/d2q9_exp/%
	@echo "  CP         $@ (model)"
	@cp $< $@

standalone/d2q9_exp/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d2q9_exp and destination package/src

package/src/d2q9_exp/%:$(SRC)/%.Rt $(SRC)/d2q9_exp/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_exp -o $@ $(RTOPT) || rm $@

package/src/d2q9_exp/%:$(SRC)/d2q9_exp/%.Rt $(SRC)/d2q9_exp/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_exp -o $@ $(RTOPT) || rm $@

package/src/d2q9_exp/%:$(SRC)/d2q9_exp/%
	@echo "  CP         $@ (model)"
	@cp $< $@

package/src/d2q9_exp/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d2q9_heat and destination standalone

standalone/d2q9_heat/%:$(SRC)/%.Rt $(SRC)/d2q9_heat/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_heat -o $@ $(RTOPT) || rm $@

standalone/d2q9_heat/%:$(SRC)/d2q9_heat/%.Rt $(SRC)/d2q9_heat/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_heat -o $@ $(RTOPT) || rm $@

standalone/d2q9_heat/%:$(SRC)/d2q9_heat/%
	@echo "  CP         $@ (model)"
	@cp $< $@

standalone/d2q9_heat/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d2q9_heat and destination package/src

package/src/d2q9_heat/%:$(SRC)/%.Rt $(SRC)/d2q9_heat/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_heat -o $@ $(RTOPT) || rm $@

package/src/d2q9_heat/%:$(SRC)/d2q9_heat/%.Rt $(SRC)/d2q9_heat/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_heat -o $@ $(RTOPT) || rm $@

package/src/d2q9_heat/%:$(SRC)/d2q9_heat/%
	@echo "  CP         $@ (model)"
	@cp $< $@

package/src/d2q9_heat/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d2q9_heat_adj and destination standalone

standalone/d2q9_heat_adj/%:$(SRC)/%.Rt $(SRC)/d2q9_heat_adj/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_heat_adj -o $@ $(RTOPT) || rm $@

standalone/d2q9_heat_adj/%:$(SRC)/d2q9_heat_adj/%.Rt $(SRC)/d2q9_heat_adj/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_heat_adj -o $@ $(RTOPT) || rm $@

standalone/d2q9_heat_adj/%:$(SRC)/d2q9_heat_adj/%
	@echo "  CP         $@ (model)"
	@cp $< $@

standalone/d2q9_heat_adj/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d2q9_heat_adj and destination package/src

package/src/d2q9_heat_adj/%:$(SRC)/%.Rt $(SRC)/d2q9_heat_adj/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_heat_adj -o $@ $(RTOPT) || rm $@

package/src/d2q9_heat_adj/%:$(SRC)/d2q9_heat_adj/%.Rt $(SRC)/d2q9_heat_adj/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_heat_adj -o $@ $(RTOPT) || rm $@

package/src/d2q9_heat_adj/%:$(SRC)/d2q9_heat_adj/%
	@echo "  CP         $@ (model)"
	@cp $< $@

package/src/d2q9_heat_adj/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d2q9_heat_diff and destination standalone

standalone/d2q9_heat_diff/%:$(SRC)/%.Rt $(SRC)/d2q9_heat_diff/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_heat_diff -o $@ $(RTOPT) || rm $@

standalone/d2q9_heat_diff/%:$(SRC)/d2q9_heat_diff/%.Rt $(SRC)/d2q9_heat_diff/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_heat_diff -o $@ $(RTOPT) || rm $@

standalone/d2q9_heat_diff/%:$(SRC)/d2q9_heat_diff/%
	@echo "  CP         $@ (model)"
	@cp $< $@

standalone/d2q9_heat_diff/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d2q9_heat_diff and destination package/src

package/src/d2q9_heat_diff/%:$(SRC)/%.Rt $(SRC)/d2q9_heat_diff/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_heat_diff -o $@ $(RTOPT) || rm $@

package/src/d2q9_heat_diff/%:$(SRC)/d2q9_heat_diff/%.Rt $(SRC)/d2q9_heat_diff/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_heat_diff -o $@ $(RTOPT) || rm $@

package/src/d2q9_heat_diff/%:$(SRC)/d2q9_heat_diff/%
	@echo "  CP         $@ (model)"
	@cp $< $@

package/src/d2q9_heat_diff/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d2q9_kuper and destination standalone

standalone/d2q9_kuper/%:$(SRC)/%.Rt $(SRC)/d2q9_kuper/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_kuper -o $@ $(RTOPT) || rm $@

standalone/d2q9_kuper/%:$(SRC)/d2q9_kuper/%.Rt $(SRC)/d2q9_kuper/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_kuper -o $@ $(RTOPT) || rm $@

standalone/d2q9_kuper/%:$(SRC)/d2q9_kuper/%
	@echo "  CP         $@ (model)"
	@cp $< $@

standalone/d2q9_kuper/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d2q9_kuper and destination package/src

package/src/d2q9_kuper/%:$(SRC)/%.Rt $(SRC)/d2q9_kuper/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_kuper -o $@ $(RTOPT) || rm $@

package/src/d2q9_kuper/%:$(SRC)/d2q9_kuper/%.Rt $(SRC)/d2q9_kuper/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_kuper -o $@ $(RTOPT) || rm $@

package/src/d2q9_kuper/%:$(SRC)/d2q9_kuper/%
	@echo "  CP         $@ (model)"
	@cp $< $@

package/src/d2q9_kuper/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d2q9_kuper2 and destination standalone

standalone/d2q9_kuper2/%:$(SRC)/%.Rt $(SRC)/d2q9_kuper2/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_kuper2 -o $@ $(RTOPT) || rm $@

standalone/d2q9_kuper2/%:$(SRC)/d2q9_kuper2/%.Rt $(SRC)/d2q9_kuper2/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_kuper2 -o $@ $(RTOPT) || rm $@

standalone/d2q9_kuper2/%:$(SRC)/d2q9_kuper2/%
	@echo "  CP         $@ (model)"
	@cp $< $@

standalone/d2q9_kuper2/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d2q9_kuper2 and destination package/src

package/src/d2q9_kuper2/%:$(SRC)/%.Rt $(SRC)/d2q9_kuper2/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_kuper2 -o $@ $(RTOPT) || rm $@

package/src/d2q9_kuper2/%:$(SRC)/d2q9_kuper2/%.Rt $(SRC)/d2q9_kuper2/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_kuper2 -o $@ $(RTOPT) || rm $@

package/src/d2q9_kuper2/%:$(SRC)/d2q9_kuper2/%
	@echo "  CP         $@ (model)"
	@cp $< $@

package/src/d2q9_kuper2/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d2q9_mes and destination standalone

standalone/d2q9_mes/%:$(SRC)/%.Rt $(SRC)/d2q9_mes/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_mes -o $@ $(RTOPT) || rm $@

standalone/d2q9_mes/%:$(SRC)/d2q9_mes/%.Rt $(SRC)/d2q9_mes/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_mes -o $@ $(RTOPT) || rm $@

standalone/d2q9_mes/%:$(SRC)/d2q9_mes/%
	@echo "  CP         $@ (model)"
	@cp $< $@

standalone/d2q9_mes/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d2q9_mes and destination package/src

package/src/d2q9_mes/%:$(SRC)/%.Rt $(SRC)/d2q9_mes/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_mes -o $@ $(RTOPT) || rm $@

package/src/d2q9_mes/%:$(SRC)/d2q9_mes/%.Rt $(SRC)/d2q9_mes/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_mes -o $@ $(RTOPT) || rm $@

package/src/d2q9_mes/%:$(SRC)/d2q9_mes/%
	@echo "  CP         $@ (model)"
	@cp $< $@

package/src/d2q9_mes/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d2q9_rho and destination standalone

standalone/d2q9_rho/%:$(SRC)/%.Rt $(SRC)/d2q9_rho/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_rho -o $@ $(RTOPT) || rm $@

standalone/d2q9_rho/%:$(SRC)/d2q9_rho/%.Rt $(SRC)/d2q9_rho/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_rho -o $@ $(RTOPT) || rm $@

standalone/d2q9_rho/%:$(SRC)/d2q9_rho/%
	@echo "  CP         $@ (model)"
	@cp $< $@

standalone/d2q9_rho/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d2q9_rho and destination package/src

package/src/d2q9_rho/%:$(SRC)/%.Rt $(SRC)/d2q9_rho/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_rho -o $@ $(RTOPT) || rm $@

package/src/d2q9_rho/%:$(SRC)/d2q9_rho/%.Rt $(SRC)/d2q9_rho/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_rho -o $@ $(RTOPT) || rm $@

package/src/d2q9_rho/%:$(SRC)/d2q9_rho/%
	@echo "  CP         $@ (model)"
	@cp $< $@

package/src/d2q9_rho/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d2q9_transport and destination standalone

standalone/d2q9_transport/%:$(SRC)/%.Rt $(SRC)/d2q9_transport/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_transport -o $@ $(RTOPT) || rm $@

standalone/d2q9_transport/%:$(SRC)/d2q9_transport/%.Rt $(SRC)/d2q9_transport/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_transport -o $@ $(RTOPT) || rm $@

standalone/d2q9_transport/%:$(SRC)/d2q9_transport/%
	@echo "  CP         $@ (model)"
	@cp $< $@

standalone/d2q9_transport/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d2q9_transport and destination package/src

package/src/d2q9_transport/%:$(SRC)/%.Rt $(SRC)/d2q9_transport/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_transport -o $@ $(RTOPT) || rm $@

package/src/d2q9_transport/%:$(SRC)/d2q9_transport/%.Rt $(SRC)/d2q9_transport/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_transport -o $@ $(RTOPT) || rm $@

package/src/d2q9_transport/%:$(SRC)/d2q9_transport/%
	@echo "  CP         $@ (model)"
	@cp $< $@

package/src/d2q9_transport/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d2q9_transport_adj and destination standalone

standalone/d2q9_transport_adj/%:$(SRC)/%.Rt $(SRC)/d2q9_transport_adj/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_transport_adj -o $@ $(RTOPT) || rm $@

standalone/d2q9_transport_adj/%:$(SRC)/d2q9_transport_adj/%.Rt $(SRC)/d2q9_transport_adj/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_transport_adj -o $@ $(RTOPT) || rm $@

standalone/d2q9_transport_adj/%:$(SRC)/d2q9_transport_adj/%
	@echo "  CP         $@ (model)"
	@cp $< $@

standalone/d2q9_transport_adj/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d2q9_transport_adj and destination package/src

package/src/d2q9_transport_adj/%:$(SRC)/%.Rt $(SRC)/d2q9_transport_adj/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_transport_adj -o $@ $(RTOPT) || rm $@

package/src/d2q9_transport_adj/%:$(SRC)/d2q9_transport_adj/%.Rt $(SRC)/d2q9_transport_adj/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_transport_adj -o $@ $(RTOPT) || rm $@

package/src/d2q9_transport_adj/%:$(SRC)/d2q9_transport_adj/%
	@echo "  CP         $@ (model)"
	@cp $< $@

package/src/d2q9_transport_adj/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d2q9_two and destination standalone

standalone/d2q9_two/%:$(SRC)/%.Rt $(SRC)/d2q9_two/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_two -o $@ $(RTOPT) || rm $@

standalone/d2q9_two/%:$(SRC)/d2q9_two/%.Rt $(SRC)/d2q9_two/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_two -o $@ $(RTOPT) || rm $@

standalone/d2q9_two/%:$(SRC)/d2q9_two/%
	@echo "  CP         $@ (model)"
	@cp $< $@

standalone/d2q9_two/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d2q9_two and destination package/src

package/src/d2q9_two/%:$(SRC)/%.Rt $(SRC)/d2q9_two/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_two -o $@ $(RTOPT) || rm $@

package/src/d2q9_two/%:$(SRC)/d2q9_two/%.Rt $(SRC)/d2q9_two/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_two -o $@ $(RTOPT) || rm $@

package/src/d2q9_two/%:$(SRC)/d2q9_two/%
	@echo "  CP         $@ (model)"
	@cp $< $@

package/src/d2q9_two/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d3q19 and destination standalone

standalone/d3q19/%:$(SRC)/%.Rt $(SRC)/d3q19/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d3q19 -o $@ $(RTOPT) || rm $@

standalone/d3q19/%:$(SRC)/d3q19/%.Rt $(SRC)/d3q19/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d3q19 -o $@ $(RTOPT) || rm $@

standalone/d3q19/%:$(SRC)/d3q19/%
	@echo "  CP         $@ (model)"
	@cp $< $@

standalone/d3q19/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d3q19 and destination package/src

package/src/d3q19/%:$(SRC)/%.Rt $(SRC)/d3q19/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d3q19 -o $@ $(RTOPT) || rm $@

package/src/d3q19/%:$(SRC)/d3q19/%.Rt $(SRC)/d3q19/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d3q19 -o $@ $(RTOPT) || rm $@

package/src/d3q19/%:$(SRC)/d3q19/%
	@echo "  CP         $@ (model)"
	@cp $< $@

package/src/d3q19/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d3q19_adj and destination standalone

standalone/d3q19_adj/%:$(SRC)/%.Rt $(SRC)/d3q19_adj/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d3q19_adj -o $@ $(RTOPT) || rm $@

standalone/d3q19_adj/%:$(SRC)/d3q19_adj/%.Rt $(SRC)/d3q19_adj/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d3q19_adj -o $@ $(RTOPT) || rm $@

standalone/d3q19_adj/%:$(SRC)/d3q19_adj/%
	@echo "  CP         $@ (model)"
	@cp $< $@

standalone/d3q19_adj/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d3q19_adj and destination package/src

package/src/d3q19_adj/%:$(SRC)/%.Rt $(SRC)/d3q19_adj/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d3q19_adj -o $@ $(RTOPT) || rm $@

package/src/d3q19_adj/%:$(SRC)/d3q19_adj/%.Rt $(SRC)/d3q19_adj/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d3q19_adj -o $@ $(RTOPT) || rm $@

package/src/d3q19_adj/%:$(SRC)/d3q19_adj/%
	@echo "  CP         $@ (model)"
	@cp $< $@

package/src/d3q19_adj/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d3q19_heat and destination standalone

standalone/d3q19_heat/%:$(SRC)/%.Rt $(SRC)/d3q19_heat/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d3q19_heat -o $@ $(RTOPT) || rm $@

standalone/d3q19_heat/%:$(SRC)/d3q19_heat/%.Rt $(SRC)/d3q19_heat/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d3q19_heat -o $@ $(RTOPT) || rm $@

standalone/d3q19_heat/%:$(SRC)/d3q19_heat/%
	@echo "  CP         $@ (model)"
	@cp $< $@

standalone/d3q19_heat/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d3q19_heat and destination package/src

package/src/d3q19_heat/%:$(SRC)/%.Rt $(SRC)/d3q19_heat/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d3q19_heat -o $@ $(RTOPT) || rm $@

package/src/d3q19_heat/%:$(SRC)/d3q19_heat/%.Rt $(SRC)/d3q19_heat/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d3q19_heat -o $@ $(RTOPT) || rm $@

package/src/d3q19_heat/%:$(SRC)/d3q19_heat/%
	@echo "  CP         $@ (model)"
	@cp $< $@

package/src/d3q19_heat/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d3q19_heat_adj and destination standalone

standalone/d3q19_heat_adj/%:$(SRC)/%.Rt $(SRC)/d3q19_heat_adj/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d3q19_heat_adj -o $@ $(RTOPT) || rm $@

standalone/d3q19_heat_adj/%:$(SRC)/d3q19_heat_adj/%.Rt $(SRC)/d3q19_heat_adj/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d3q19_heat_adj -o $@ $(RTOPT) || rm $@

standalone/d3q19_heat_adj/%:$(SRC)/d3q19_heat_adj/%
	@echo "  CP         $@ (model)"
	@cp $< $@

standalone/d3q19_heat_adj/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d3q19_heat_adj and destination package/src

package/src/d3q19_heat_adj/%:$(SRC)/%.Rt $(SRC)/d3q19_heat_adj/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d3q19_heat_adj -o $@ $(RTOPT) || rm $@

package/src/d3q19_heat_adj/%:$(SRC)/d3q19_heat_adj/%.Rt $(SRC)/d3q19_heat_adj/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d3q19_heat_adj -o $@ $(RTOPT) || rm $@

package/src/d3q19_heat_adj/%:$(SRC)/d3q19_heat_adj/%
	@echo "  CP         $@ (model)"
	@cp $< $@

package/src/d3q19_heat_adj/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d3q19_n and destination standalone

standalone/d3q19_n/%:$(SRC)/%.Rt $(SRC)/d3q19_n/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d3q19_n -o $@ $(RTOPT) || rm $@

standalone/d3q19_n/%:$(SRC)/d3q19_n/%.Rt $(SRC)/d3q19_n/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d3q19_n -o $@ $(RTOPT) || rm $@

standalone/d3q19_n/%:$(SRC)/d3q19_n/%
	@echo "  CP         $@ (model)"
	@cp $< $@

standalone/d3q19_n/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d3q19_n and destination package/src

package/src/d3q19_n/%:$(SRC)/%.Rt $(SRC)/d3q19_n/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d3q19_n -o $@ $(RTOPT) || rm $@

package/src/d3q19_n/%:$(SRC)/d3q19_n/%.Rt $(SRC)/d3q19_n/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d3q19_n -o $@ $(RTOPT) || rm $@

package/src/d3q19_n/%:$(SRC)/d3q19_n/%
	@echo "  CP         $@ (model)"
	@cp $< $@

package/src/d3q19_n/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@




%.hp:%.c
	@echo "MAKEHEADERS  $<"
	@cp $< $<.cpp; $(MAKEHEADERS) $<.cpp && sed 's/extern//' $<.hpp > $@
	@rm $<.cpp $<.hpp

%.hp:%.cpp
	@echo "MAKEHEADERS  $<"
	@cp $< $<.cpp; $(MAKEHEADERS) $<.cpp && sed 's/extern//' $<.hpp > $@
	@rm $<.cpp $<.hpp

