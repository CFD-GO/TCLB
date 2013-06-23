
# MODELS: 
#	d1q3, d2q9, d2q9_adj, d2q9_adj_smooth, d2q9_adj_top, d2q9_entropic, d2q9_exp, d2q9_heat, d2q9_heat_adj, d2q9_kuper, d3q19, d3q19_adj, d3q19_heat, d3q19_heat_adj

ADJOINT=0        # calculate adjoint: 1-on, 0-off
GRAPHICS=1       # GLUT graphics: 1-on, 0-off
GRID3D=0         # use 3D block grid (only avaliable on capability 2.x): 1-on, 0-off
ARCH=sm_11       # CUDA architecture: sm_10 for capability 1.0, sm_13 for capability 1.3 etc.
DOUBLE=0         # precision: 1-double, 0-float


#######################################################################################################################

all: d1q3 d2q9 d2q9_adj d2q9_adj_smooth d2q9_adj_top d2q9_entropic d2q9_exp d2q9_heat d2q9_heat_adj d2q9_kuper d3q19 d3q19_adj d3q19_heat d3q19_heat_adj

.PHONY: all clean dummy

makefile:src/makefile.main.Rt src/*
	@tools/RT -q -f $< -o $@

#Rpackage : source package/configure
#	R CMD build package
	
#package/configure:package/configure.ac
#	@echo "AUTOCONF     $@"
#	@cd package; autoconf

RT = tools/RT
RS = R  --slave --quiet --vanilla -f
ADMOD = tools/ADmod.R
MAKEHEADERS = tools/makeheaders

SRC=src

#ifeq '$(strip $(STANDALONE))' '1'
 DEST=standalone
 ADDITIONALS=makefile model README.md dep.mk
 SOURCE_CU+=main.cu
 HEADERS_H+=DataLine.h
#else
# DEST=package/src
# ADDITIONALS=package/src/Makefile.in package/data/LB.RData
#endif

SOURCE_CU+=Global.cu Lattice.cu vtkLattice.cu vtkOutput.cu cross.cu cuda.cu LatticeContainer.cu Dynamics.c inter.cpp Solver.cpp pugixml.cpp Geometry.cu def.cpp unit.cpp
SOURCE_R=conf.R Dynamics.R
SOURCE=$(addprefix $(DEST)/,$(SOURCE_CU))
HEADERS_H+=Global.h gpu_anim.h LatticeContainer.h Lattice.h Node.h Region.h vtkLattice.h vtkOutput.h cross.h gl_helper.h Dynamics.h Dynamics.hp types.h Node_types.h Solver.h pugixml.hpp pugiconfig.hpp Geometry.h def.h utils.h unit.h
HEADERS=$(addprefix $(DEST)/,$(HEADERS_H))

ALL_FILES=$(SOURCE_CU) $(HEADERS_H) $(ADDITIONALS)
DEST_FILES=$(addprefix $(DEST)/,$(ALL_FILES))

AOUT=main

CC=nvcc
CCTXT=NVCC

RTOPT=

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


###############################################################################
######       AUTO GENERATED CASES for RT                                 ######
###############################################################################


d1q3:  | standalone/d1q3

d1q3: standalone/d1q3 standalone/d1q3/main
	@echo "  DONE       $@"
standalone/d1q3/main:standalone/d1q3 $(addprefix standalone/d1q3/,$(ALL_FILES))
	@cd standalone/d1q3; $(MAKE)

standalone/d1q3 :
	mkdir -p $@

d2q9:  | standalone/d2q9

d2q9: standalone/d2q9 standalone/d2q9/main
	@echo "  DONE       $@"
standalone/d2q9/main:standalone/d2q9 $(addprefix standalone/d2q9/,$(ALL_FILES))
	@cd standalone/d2q9; $(MAKE)

standalone/d2q9 :
	mkdir -p $@

d2q9_adj:  | standalone/d2q9_adj

d2q9_adj: standalone/d2q9_adj standalone/d2q9_adj/main
	@echo "  DONE       $@"
standalone/d2q9_adj/main:standalone/d2q9_adj $(addprefix standalone/d2q9_adj/,$(ALL_FILES))
	@cd standalone/d2q9_adj; $(MAKE)

standalone/d2q9_adj :
	mkdir -p $@

d2q9_adj_smooth:  | standalone/d2q9_adj_smooth

d2q9_adj_smooth: standalone/d2q9_adj_smooth standalone/d2q9_adj_smooth/main
	@echo "  DONE       $@"
standalone/d2q9_adj_smooth/main:standalone/d2q9_adj_smooth $(addprefix standalone/d2q9_adj_smooth/,$(ALL_FILES))
	@cd standalone/d2q9_adj_smooth; $(MAKE)

standalone/d2q9_adj_smooth :
	mkdir -p $@

d2q9_adj_top:  | standalone/d2q9_adj_top

d2q9_adj_top: standalone/d2q9_adj_top standalone/d2q9_adj_top/main
	@echo "  DONE       $@"
standalone/d2q9_adj_top/main:standalone/d2q9_adj_top $(addprefix standalone/d2q9_adj_top/,$(ALL_FILES))
	@cd standalone/d2q9_adj_top; $(MAKE)

standalone/d2q9_adj_top :
	mkdir -p $@

d2q9_entropic:  | standalone/d2q9_entropic

d2q9_entropic: standalone/d2q9_entropic standalone/d2q9_entropic/main
	@echo "  DONE       $@"
standalone/d2q9_entropic/main:standalone/d2q9_entropic $(addprefix standalone/d2q9_entropic/,$(ALL_FILES))
	@cd standalone/d2q9_entropic; $(MAKE)

standalone/d2q9_entropic :
	mkdir -p $@

d2q9_exp:  | standalone/d2q9_exp

d2q9_exp: standalone/d2q9_exp standalone/d2q9_exp/main
	@echo "  DONE       $@"
standalone/d2q9_exp/main:standalone/d2q9_exp $(addprefix standalone/d2q9_exp/,$(ALL_FILES))
	@cd standalone/d2q9_exp; $(MAKE)

standalone/d2q9_exp :
	mkdir -p $@

d2q9_heat:  | standalone/d2q9_heat

d2q9_heat: standalone/d2q9_heat standalone/d2q9_heat/main
	@echo "  DONE       $@"
standalone/d2q9_heat/main:standalone/d2q9_heat $(addprefix standalone/d2q9_heat/,$(ALL_FILES))
	@cd standalone/d2q9_heat; $(MAKE)

standalone/d2q9_heat :
	mkdir -p $@

d2q9_heat_adj:  | standalone/d2q9_heat_adj

d2q9_heat_adj: standalone/d2q9_heat_adj standalone/d2q9_heat_adj/main
	@echo "  DONE       $@"
standalone/d2q9_heat_adj/main:standalone/d2q9_heat_adj $(addprefix standalone/d2q9_heat_adj/,$(ALL_FILES))
	@cd standalone/d2q9_heat_adj; $(MAKE)

standalone/d2q9_heat_adj :
	mkdir -p $@

d2q9_kuper:  | standalone/d2q9_kuper

d2q9_kuper: standalone/d2q9_kuper standalone/d2q9_kuper/main
	@echo "  DONE       $@"
standalone/d2q9_kuper/main:standalone/d2q9_kuper $(addprefix standalone/d2q9_kuper/,$(ALL_FILES))
	@cd standalone/d2q9_kuper; $(MAKE)

standalone/d2q9_kuper :
	mkdir -p $@

d3q19:  | standalone/d3q19

d3q19: standalone/d3q19 standalone/d3q19/main
	@echo "  DONE       $@"
standalone/d3q19/main:standalone/d3q19 $(addprefix standalone/d3q19/,$(ALL_FILES))
	@cd standalone/d3q19; $(MAKE)

standalone/d3q19 :
	mkdir -p $@

d3q19_adj:  | standalone/d3q19_adj

d3q19_adj: standalone/d3q19_adj standalone/d3q19_adj/main
	@echo "  DONE       $@"
standalone/d3q19_adj/main:standalone/d3q19_adj $(addprefix standalone/d3q19_adj/,$(ALL_FILES))
	@cd standalone/d3q19_adj; $(MAKE)

standalone/d3q19_adj :
	mkdir -p $@

d3q19_heat:  | standalone/d3q19_heat

d3q19_heat: standalone/d3q19_heat standalone/d3q19_heat/main
	@echo "  DONE       $@"
standalone/d3q19_heat/main:standalone/d3q19_heat $(addprefix standalone/d3q19_heat/,$(ALL_FILES))
	@cd standalone/d3q19_heat; $(MAKE)

standalone/d3q19_heat :
	mkdir -p $@

d3q19_heat_adj:  | standalone/d3q19_heat_adj

d3q19_heat_adj: standalone/d3q19_heat_adj standalone/d3q19_heat_adj/main
	@echo "  DONE       $@"
standalone/d3q19_heat_adj/main:standalone/d3q19_heat_adj $(addprefix standalone/d3q19_heat_adj/,$(ALL_FILES))
	@cd standalone/d3q19_heat_adj; $(MAKE)

standalone/d3q19_heat_adj :
	mkdir -p $@




# for model d1q3 and destination standalone

standalone/d1q3/Dynamics_b.c standalone/d1q3/Dynamics_b.h standalone/d1q3/types_b.h standalone/d1q3/ADpre_.h standalone/d1q3/ADpre__b.h : standalone/d1q3/tapenade.run

.INTERMEDIATE : tapenade.run

standalone/d1q3/tapenade.run : standalone/d1q3 standalone/d1q3/Dynamics.c standalone/d1q3/ADpre.h standalone/d1q3/ADpre_b.h standalone/d1q3/Dynamics.h standalone/d1q3/Node_types.h standalone/d1q3/types.h standalone/d1q3/ADset.sh
	@echo "  TAPENADE   $<"
	@(cd $<; ../../tools/makeAD)


standalone/d1q3/dep.mk:tools/dep.R $(addprefix standalone/d1q3/,$(SOURCE_CU) $(HEADERS_H))
	@echo "  AUTO-DEP   $@"
	@cd standalone/d1q3; $(RS) ../../$<

standalone/d1q3/%:$(SRC)/%.Rt $(SRC)/d1q3/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d1q3 -o $@ $(RTOPT) MODEL=\"d1q3\" || rm $@

standalone/d1q3/%:$(SRC)/d1q3/%.Rt $(SRC)/d1q3/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d1q3 -o $@ $(RTOPT) MODEL=\"d1q3\" || rm $@

standalone/d1q3/%:$(SRC)/d1q3/%
	@echo "  CP         $@ (model)"
	@cp $< $@

standalone/d1q3/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d1q3 and destination package/src

package/src/d1q3/Dynamics_b.c package/src/d1q3/Dynamics_b.h package/src/d1q3/types_b.h package/src/d1q3/ADpre_.h package/src/d1q3/ADpre__b.h : package/src/d1q3/tapenade.run

.INTERMEDIATE : tapenade.run

package/src/d1q3/tapenade.run : package/src/d1q3 package/src/d1q3/Dynamics.c package/src/d1q3/ADpre.h package/src/d1q3/ADpre_b.h package/src/d1q3/Dynamics.h package/src/d1q3/Node_types.h package/src/d1q3/types.h package/src/d1q3/ADset.sh
	@echo "  TAPENADE   $<"
	@(cd $<; ../../tools/makeAD)


package/src/d1q3/dep.mk:tools/dep.R $(addprefix package/src/d1q3/,$(SOURCE_CU) $(HEADERS_H))
	@echo "  AUTO-DEP   $@"
	@cd package/src/d1q3; $(RS) ../../$<

package/src/d1q3/%:$(SRC)/%.Rt $(SRC)/d1q3/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d1q3 -o $@ $(RTOPT) MODEL=\"d1q3\" || rm $@

package/src/d1q3/%:$(SRC)/d1q3/%.Rt $(SRC)/d1q3/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d1q3 -o $@ $(RTOPT) MODEL=\"d1q3\" || rm $@

package/src/d1q3/%:$(SRC)/d1q3/%
	@echo "  CP         $@ (model)"
	@cp $< $@

package/src/d1q3/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d2q9 and destination standalone

standalone/d2q9/Dynamics_b.c standalone/d2q9/Dynamics_b.h standalone/d2q9/types_b.h standalone/d2q9/ADpre_.h standalone/d2q9/ADpre__b.h : standalone/d2q9/tapenade.run

.INTERMEDIATE : tapenade.run

standalone/d2q9/tapenade.run : standalone/d2q9 standalone/d2q9/Dynamics.c standalone/d2q9/ADpre.h standalone/d2q9/ADpre_b.h standalone/d2q9/Dynamics.h standalone/d2q9/Node_types.h standalone/d2q9/types.h standalone/d2q9/ADset.sh
	@echo "  TAPENADE   $<"
	@(cd $<; ../../tools/makeAD)


standalone/d2q9/dep.mk:tools/dep.R $(addprefix standalone/d2q9/,$(SOURCE_CU) $(HEADERS_H))
	@echo "  AUTO-DEP   $@"
	@cd standalone/d2q9; $(RS) ../../$<

standalone/d2q9/%:$(SRC)/%.Rt $(SRC)/d2q9/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9 -o $@ $(RTOPT) MODEL=\"d2q9\" || rm $@

standalone/d2q9/%:$(SRC)/d2q9/%.Rt $(SRC)/d2q9/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9 -o $@ $(RTOPT) MODEL=\"d2q9\" || rm $@

standalone/d2q9/%:$(SRC)/d2q9/%
	@echo "  CP         $@ (model)"
	@cp $< $@

standalone/d2q9/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d2q9 and destination package/src

package/src/d2q9/Dynamics_b.c package/src/d2q9/Dynamics_b.h package/src/d2q9/types_b.h package/src/d2q9/ADpre_.h package/src/d2q9/ADpre__b.h : package/src/d2q9/tapenade.run

.INTERMEDIATE : tapenade.run

package/src/d2q9/tapenade.run : package/src/d2q9 package/src/d2q9/Dynamics.c package/src/d2q9/ADpre.h package/src/d2q9/ADpre_b.h package/src/d2q9/Dynamics.h package/src/d2q9/Node_types.h package/src/d2q9/types.h package/src/d2q9/ADset.sh
	@echo "  TAPENADE   $<"
	@(cd $<; ../../tools/makeAD)


package/src/d2q9/dep.mk:tools/dep.R $(addprefix package/src/d2q9/,$(SOURCE_CU) $(HEADERS_H))
	@echo "  AUTO-DEP   $@"
	@cd package/src/d2q9; $(RS) ../../$<

package/src/d2q9/%:$(SRC)/%.Rt $(SRC)/d2q9/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9 -o $@ $(RTOPT) MODEL=\"d2q9\" || rm $@

package/src/d2q9/%:$(SRC)/d2q9/%.Rt $(SRC)/d2q9/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9 -o $@ $(RTOPT) MODEL=\"d2q9\" || rm $@

package/src/d2q9/%:$(SRC)/d2q9/%
	@echo "  CP         $@ (model)"
	@cp $< $@

package/src/d2q9/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d2q9_adj and destination standalone

standalone/d2q9_adj/Dynamics_b.c standalone/d2q9_adj/Dynamics_b.h standalone/d2q9_adj/types_b.h standalone/d2q9_adj/ADpre_.h standalone/d2q9_adj/ADpre__b.h : standalone/d2q9_adj/tapenade.run

.INTERMEDIATE : tapenade.run

standalone/d2q9_adj/tapenade.run : standalone/d2q9_adj standalone/d2q9_adj/Dynamics.c standalone/d2q9_adj/ADpre.h standalone/d2q9_adj/ADpre_b.h standalone/d2q9_adj/Dynamics.h standalone/d2q9_adj/Node_types.h standalone/d2q9_adj/types.h standalone/d2q9_adj/ADset.sh
	@echo "  TAPENADE   $<"
	@(cd $<; ../../tools/makeAD)


standalone/d2q9_adj/dep.mk:tools/dep.R $(addprefix standalone/d2q9_adj/,$(SOURCE_CU) $(HEADERS_H))
	@echo "  AUTO-DEP   $@"
	@cd standalone/d2q9_adj; $(RS) ../../$<

standalone/d2q9_adj/%:$(SRC)/%.Rt $(SRC)/d2q9_adj/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_adj -o $@ $(RTOPT) MODEL=\"d2q9_adj\" || rm $@

standalone/d2q9_adj/%:$(SRC)/d2q9_adj/%.Rt $(SRC)/d2q9_adj/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_adj -o $@ $(RTOPT) MODEL=\"d2q9_adj\" || rm $@

standalone/d2q9_adj/%:$(SRC)/d2q9_adj/%
	@echo "  CP         $@ (model)"
	@cp $< $@

standalone/d2q9_adj/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d2q9_adj and destination package/src

package/src/d2q9_adj/Dynamics_b.c package/src/d2q9_adj/Dynamics_b.h package/src/d2q9_adj/types_b.h package/src/d2q9_adj/ADpre_.h package/src/d2q9_adj/ADpre__b.h : package/src/d2q9_adj/tapenade.run

.INTERMEDIATE : tapenade.run

package/src/d2q9_adj/tapenade.run : package/src/d2q9_adj package/src/d2q9_adj/Dynamics.c package/src/d2q9_adj/ADpre.h package/src/d2q9_adj/ADpre_b.h package/src/d2q9_adj/Dynamics.h package/src/d2q9_adj/Node_types.h package/src/d2q9_adj/types.h package/src/d2q9_adj/ADset.sh
	@echo "  TAPENADE   $<"
	@(cd $<; ../../tools/makeAD)


package/src/d2q9_adj/dep.mk:tools/dep.R $(addprefix package/src/d2q9_adj/,$(SOURCE_CU) $(HEADERS_H))
	@echo "  AUTO-DEP   $@"
	@cd package/src/d2q9_adj; $(RS) ../../$<

package/src/d2q9_adj/%:$(SRC)/%.Rt $(SRC)/d2q9_adj/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_adj -o $@ $(RTOPT) MODEL=\"d2q9_adj\" || rm $@

package/src/d2q9_adj/%:$(SRC)/d2q9_adj/%.Rt $(SRC)/d2q9_adj/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_adj -o $@ $(RTOPT) MODEL=\"d2q9_adj\" || rm $@

package/src/d2q9_adj/%:$(SRC)/d2q9_adj/%
	@echo "  CP         $@ (model)"
	@cp $< $@

package/src/d2q9_adj/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d2q9_adj_smooth and destination standalone

standalone/d2q9_adj_smooth/Dynamics_b.c standalone/d2q9_adj_smooth/Dynamics_b.h standalone/d2q9_adj_smooth/types_b.h standalone/d2q9_adj_smooth/ADpre_.h standalone/d2q9_adj_smooth/ADpre__b.h : standalone/d2q9_adj_smooth/tapenade.run

.INTERMEDIATE : tapenade.run

standalone/d2q9_adj_smooth/tapenade.run : standalone/d2q9_adj_smooth standalone/d2q9_adj_smooth/Dynamics.c standalone/d2q9_adj_smooth/ADpre.h standalone/d2q9_adj_smooth/ADpre_b.h standalone/d2q9_adj_smooth/Dynamics.h standalone/d2q9_adj_smooth/Node_types.h standalone/d2q9_adj_smooth/types.h standalone/d2q9_adj_smooth/ADset.sh
	@echo "  TAPENADE   $<"
	@(cd $<; ../../tools/makeAD)


standalone/d2q9_adj_smooth/dep.mk:tools/dep.R $(addprefix standalone/d2q9_adj_smooth/,$(SOURCE_CU) $(HEADERS_H))
	@echo "  AUTO-DEP   $@"
	@cd standalone/d2q9_adj_smooth; $(RS) ../../$<

standalone/d2q9_adj_smooth/%:$(SRC)/%.Rt $(SRC)/d2q9_adj_smooth/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_adj_smooth -o $@ $(RTOPT) MODEL=\"d2q9_adj_smooth\" || rm $@

standalone/d2q9_adj_smooth/%:$(SRC)/d2q9_adj_smooth/%.Rt $(SRC)/d2q9_adj_smooth/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_adj_smooth -o $@ $(RTOPT) MODEL=\"d2q9_adj_smooth\" || rm $@

standalone/d2q9_adj_smooth/%:$(SRC)/d2q9_adj_smooth/%
	@echo "  CP         $@ (model)"
	@cp $< $@

standalone/d2q9_adj_smooth/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d2q9_adj_smooth and destination package/src

package/src/d2q9_adj_smooth/Dynamics_b.c package/src/d2q9_adj_smooth/Dynamics_b.h package/src/d2q9_adj_smooth/types_b.h package/src/d2q9_adj_smooth/ADpre_.h package/src/d2q9_adj_smooth/ADpre__b.h : package/src/d2q9_adj_smooth/tapenade.run

.INTERMEDIATE : tapenade.run

package/src/d2q9_adj_smooth/tapenade.run : package/src/d2q9_adj_smooth package/src/d2q9_adj_smooth/Dynamics.c package/src/d2q9_adj_smooth/ADpre.h package/src/d2q9_adj_smooth/ADpre_b.h package/src/d2q9_adj_smooth/Dynamics.h package/src/d2q9_adj_smooth/Node_types.h package/src/d2q9_adj_smooth/types.h package/src/d2q9_adj_smooth/ADset.sh
	@echo "  TAPENADE   $<"
	@(cd $<; ../../tools/makeAD)


package/src/d2q9_adj_smooth/dep.mk:tools/dep.R $(addprefix package/src/d2q9_adj_smooth/,$(SOURCE_CU) $(HEADERS_H))
	@echo "  AUTO-DEP   $@"
	@cd package/src/d2q9_adj_smooth; $(RS) ../../$<

package/src/d2q9_adj_smooth/%:$(SRC)/%.Rt $(SRC)/d2q9_adj_smooth/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_adj_smooth -o $@ $(RTOPT) MODEL=\"d2q9_adj_smooth\" || rm $@

package/src/d2q9_adj_smooth/%:$(SRC)/d2q9_adj_smooth/%.Rt $(SRC)/d2q9_adj_smooth/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_adj_smooth -o $@ $(RTOPT) MODEL=\"d2q9_adj_smooth\" || rm $@

package/src/d2q9_adj_smooth/%:$(SRC)/d2q9_adj_smooth/%
	@echo "  CP         $@ (model)"
	@cp $< $@

package/src/d2q9_adj_smooth/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d2q9_adj_top and destination standalone

standalone/d2q9_adj_top/Dynamics_b.c standalone/d2q9_adj_top/Dynamics_b.h standalone/d2q9_adj_top/types_b.h standalone/d2q9_adj_top/ADpre_.h standalone/d2q9_adj_top/ADpre__b.h : standalone/d2q9_adj_top/tapenade.run

.INTERMEDIATE : tapenade.run

standalone/d2q9_adj_top/tapenade.run : standalone/d2q9_adj_top standalone/d2q9_adj_top/Dynamics.c standalone/d2q9_adj_top/ADpre.h standalone/d2q9_adj_top/ADpre_b.h standalone/d2q9_adj_top/Dynamics.h standalone/d2q9_adj_top/Node_types.h standalone/d2q9_adj_top/types.h standalone/d2q9_adj_top/ADset.sh
	@echo "  TAPENADE   $<"
	@(cd $<; ../../tools/makeAD)


standalone/d2q9_adj_top/dep.mk:tools/dep.R $(addprefix standalone/d2q9_adj_top/,$(SOURCE_CU) $(HEADERS_H))
	@echo "  AUTO-DEP   $@"
	@cd standalone/d2q9_adj_top; $(RS) ../../$<

standalone/d2q9_adj_top/%:$(SRC)/%.Rt $(SRC)/d2q9_adj_top/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_adj_top -o $@ $(RTOPT) MODEL=\"d2q9_adj_top\" || rm $@

standalone/d2q9_adj_top/%:$(SRC)/d2q9_adj_top/%.Rt $(SRC)/d2q9_adj_top/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_adj_top -o $@ $(RTOPT) MODEL=\"d2q9_adj_top\" || rm $@

standalone/d2q9_adj_top/%:$(SRC)/d2q9_adj_top/%
	@echo "  CP         $@ (model)"
	@cp $< $@

standalone/d2q9_adj_top/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d2q9_adj_top and destination package/src

package/src/d2q9_adj_top/Dynamics_b.c package/src/d2q9_adj_top/Dynamics_b.h package/src/d2q9_adj_top/types_b.h package/src/d2q9_adj_top/ADpre_.h package/src/d2q9_adj_top/ADpre__b.h : package/src/d2q9_adj_top/tapenade.run

.INTERMEDIATE : tapenade.run

package/src/d2q9_adj_top/tapenade.run : package/src/d2q9_adj_top package/src/d2q9_adj_top/Dynamics.c package/src/d2q9_adj_top/ADpre.h package/src/d2q9_adj_top/ADpre_b.h package/src/d2q9_adj_top/Dynamics.h package/src/d2q9_adj_top/Node_types.h package/src/d2q9_adj_top/types.h package/src/d2q9_adj_top/ADset.sh
	@echo "  TAPENADE   $<"
	@(cd $<; ../../tools/makeAD)


package/src/d2q9_adj_top/dep.mk:tools/dep.R $(addprefix package/src/d2q9_adj_top/,$(SOURCE_CU) $(HEADERS_H))
	@echo "  AUTO-DEP   $@"
	@cd package/src/d2q9_adj_top; $(RS) ../../$<

package/src/d2q9_adj_top/%:$(SRC)/%.Rt $(SRC)/d2q9_adj_top/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_adj_top -o $@ $(RTOPT) MODEL=\"d2q9_adj_top\" || rm $@

package/src/d2q9_adj_top/%:$(SRC)/d2q9_adj_top/%.Rt $(SRC)/d2q9_adj_top/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_adj_top -o $@ $(RTOPT) MODEL=\"d2q9_adj_top\" || rm $@

package/src/d2q9_adj_top/%:$(SRC)/d2q9_adj_top/%
	@echo "  CP         $@ (model)"
	@cp $< $@

package/src/d2q9_adj_top/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d2q9_entropic and destination standalone

standalone/d2q9_entropic/Dynamics_b.c standalone/d2q9_entropic/Dynamics_b.h standalone/d2q9_entropic/types_b.h standalone/d2q9_entropic/ADpre_.h standalone/d2q9_entropic/ADpre__b.h : standalone/d2q9_entropic/tapenade.run

.INTERMEDIATE : tapenade.run

standalone/d2q9_entropic/tapenade.run : standalone/d2q9_entropic standalone/d2q9_entropic/Dynamics.c standalone/d2q9_entropic/ADpre.h standalone/d2q9_entropic/ADpre_b.h standalone/d2q9_entropic/Dynamics.h standalone/d2q9_entropic/Node_types.h standalone/d2q9_entropic/types.h standalone/d2q9_entropic/ADset.sh
	@echo "  TAPENADE   $<"
	@(cd $<; ../../tools/makeAD)


standalone/d2q9_entropic/dep.mk:tools/dep.R $(addprefix standalone/d2q9_entropic/,$(SOURCE_CU) $(HEADERS_H))
	@echo "  AUTO-DEP   $@"
	@cd standalone/d2q9_entropic; $(RS) ../../$<

standalone/d2q9_entropic/%:$(SRC)/%.Rt $(SRC)/d2q9_entropic/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_entropic -o $@ $(RTOPT) MODEL=\"d2q9_entropic\" || rm $@

standalone/d2q9_entropic/%:$(SRC)/d2q9_entropic/%.Rt $(SRC)/d2q9_entropic/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_entropic -o $@ $(RTOPT) MODEL=\"d2q9_entropic\" || rm $@

standalone/d2q9_entropic/%:$(SRC)/d2q9_entropic/%
	@echo "  CP         $@ (model)"
	@cp $< $@

standalone/d2q9_entropic/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d2q9_entropic and destination package/src

package/src/d2q9_entropic/Dynamics_b.c package/src/d2q9_entropic/Dynamics_b.h package/src/d2q9_entropic/types_b.h package/src/d2q9_entropic/ADpre_.h package/src/d2q9_entropic/ADpre__b.h : package/src/d2q9_entropic/tapenade.run

.INTERMEDIATE : tapenade.run

package/src/d2q9_entropic/tapenade.run : package/src/d2q9_entropic package/src/d2q9_entropic/Dynamics.c package/src/d2q9_entropic/ADpre.h package/src/d2q9_entropic/ADpre_b.h package/src/d2q9_entropic/Dynamics.h package/src/d2q9_entropic/Node_types.h package/src/d2q9_entropic/types.h package/src/d2q9_entropic/ADset.sh
	@echo "  TAPENADE   $<"
	@(cd $<; ../../tools/makeAD)


package/src/d2q9_entropic/dep.mk:tools/dep.R $(addprefix package/src/d2q9_entropic/,$(SOURCE_CU) $(HEADERS_H))
	@echo "  AUTO-DEP   $@"
	@cd package/src/d2q9_entropic; $(RS) ../../$<

package/src/d2q9_entropic/%:$(SRC)/%.Rt $(SRC)/d2q9_entropic/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_entropic -o $@ $(RTOPT) MODEL=\"d2q9_entropic\" || rm $@

package/src/d2q9_entropic/%:$(SRC)/d2q9_entropic/%.Rt $(SRC)/d2q9_entropic/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_entropic -o $@ $(RTOPT) MODEL=\"d2q9_entropic\" || rm $@

package/src/d2q9_entropic/%:$(SRC)/d2q9_entropic/%
	@echo "  CP         $@ (model)"
	@cp $< $@

package/src/d2q9_entropic/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d2q9_exp and destination standalone

standalone/d2q9_exp/Dynamics_b.c standalone/d2q9_exp/Dynamics_b.h standalone/d2q9_exp/types_b.h standalone/d2q9_exp/ADpre_.h standalone/d2q9_exp/ADpre__b.h : standalone/d2q9_exp/tapenade.run

.INTERMEDIATE : tapenade.run

standalone/d2q9_exp/tapenade.run : standalone/d2q9_exp standalone/d2q9_exp/Dynamics.c standalone/d2q9_exp/ADpre.h standalone/d2q9_exp/ADpre_b.h standalone/d2q9_exp/Dynamics.h standalone/d2q9_exp/Node_types.h standalone/d2q9_exp/types.h standalone/d2q9_exp/ADset.sh
	@echo "  TAPENADE   $<"
	@(cd $<; ../../tools/makeAD)


standalone/d2q9_exp/dep.mk:tools/dep.R $(addprefix standalone/d2q9_exp/,$(SOURCE_CU) $(HEADERS_H))
	@echo "  AUTO-DEP   $@"
	@cd standalone/d2q9_exp; $(RS) ../../$<

standalone/d2q9_exp/%:$(SRC)/%.Rt $(SRC)/d2q9_exp/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_exp -o $@ $(RTOPT) MODEL=\"d2q9_exp\" || rm $@

standalone/d2q9_exp/%:$(SRC)/d2q9_exp/%.Rt $(SRC)/d2q9_exp/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_exp -o $@ $(RTOPT) MODEL=\"d2q9_exp\" || rm $@

standalone/d2q9_exp/%:$(SRC)/d2q9_exp/%
	@echo "  CP         $@ (model)"
	@cp $< $@

standalone/d2q9_exp/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d2q9_exp and destination package/src

package/src/d2q9_exp/Dynamics_b.c package/src/d2q9_exp/Dynamics_b.h package/src/d2q9_exp/types_b.h package/src/d2q9_exp/ADpre_.h package/src/d2q9_exp/ADpre__b.h : package/src/d2q9_exp/tapenade.run

.INTERMEDIATE : tapenade.run

package/src/d2q9_exp/tapenade.run : package/src/d2q9_exp package/src/d2q9_exp/Dynamics.c package/src/d2q9_exp/ADpre.h package/src/d2q9_exp/ADpre_b.h package/src/d2q9_exp/Dynamics.h package/src/d2q9_exp/Node_types.h package/src/d2q9_exp/types.h package/src/d2q9_exp/ADset.sh
	@echo "  TAPENADE   $<"
	@(cd $<; ../../tools/makeAD)


package/src/d2q9_exp/dep.mk:tools/dep.R $(addprefix package/src/d2q9_exp/,$(SOURCE_CU) $(HEADERS_H))
	@echo "  AUTO-DEP   $@"
	@cd package/src/d2q9_exp; $(RS) ../../$<

package/src/d2q9_exp/%:$(SRC)/%.Rt $(SRC)/d2q9_exp/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_exp -o $@ $(RTOPT) MODEL=\"d2q9_exp\" || rm $@

package/src/d2q9_exp/%:$(SRC)/d2q9_exp/%.Rt $(SRC)/d2q9_exp/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_exp -o $@ $(RTOPT) MODEL=\"d2q9_exp\" || rm $@

package/src/d2q9_exp/%:$(SRC)/d2q9_exp/%
	@echo "  CP         $@ (model)"
	@cp $< $@

package/src/d2q9_exp/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d2q9_heat and destination standalone

standalone/d2q9_heat/Dynamics_b.c standalone/d2q9_heat/Dynamics_b.h standalone/d2q9_heat/types_b.h standalone/d2q9_heat/ADpre_.h standalone/d2q9_heat/ADpre__b.h : standalone/d2q9_heat/tapenade.run

.INTERMEDIATE : tapenade.run

standalone/d2q9_heat/tapenade.run : standalone/d2q9_heat standalone/d2q9_heat/Dynamics.c standalone/d2q9_heat/ADpre.h standalone/d2q9_heat/ADpre_b.h standalone/d2q9_heat/Dynamics.h standalone/d2q9_heat/Node_types.h standalone/d2q9_heat/types.h standalone/d2q9_heat/ADset.sh
	@echo "  TAPENADE   $<"
	@(cd $<; ../../tools/makeAD)


standalone/d2q9_heat/dep.mk:tools/dep.R $(addprefix standalone/d2q9_heat/,$(SOURCE_CU) $(HEADERS_H))
	@echo "  AUTO-DEP   $@"
	@cd standalone/d2q9_heat; $(RS) ../../$<

standalone/d2q9_heat/%:$(SRC)/%.Rt $(SRC)/d2q9_heat/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_heat -o $@ $(RTOPT) MODEL=\"d2q9_heat\" || rm $@

standalone/d2q9_heat/%:$(SRC)/d2q9_heat/%.Rt $(SRC)/d2q9_heat/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_heat -o $@ $(RTOPT) MODEL=\"d2q9_heat\" || rm $@

standalone/d2q9_heat/%:$(SRC)/d2q9_heat/%
	@echo "  CP         $@ (model)"
	@cp $< $@

standalone/d2q9_heat/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d2q9_heat and destination package/src

package/src/d2q9_heat/Dynamics_b.c package/src/d2q9_heat/Dynamics_b.h package/src/d2q9_heat/types_b.h package/src/d2q9_heat/ADpre_.h package/src/d2q9_heat/ADpre__b.h : package/src/d2q9_heat/tapenade.run

.INTERMEDIATE : tapenade.run

package/src/d2q9_heat/tapenade.run : package/src/d2q9_heat package/src/d2q9_heat/Dynamics.c package/src/d2q9_heat/ADpre.h package/src/d2q9_heat/ADpre_b.h package/src/d2q9_heat/Dynamics.h package/src/d2q9_heat/Node_types.h package/src/d2q9_heat/types.h package/src/d2q9_heat/ADset.sh
	@echo "  TAPENADE   $<"
	@(cd $<; ../../tools/makeAD)


package/src/d2q9_heat/dep.mk:tools/dep.R $(addprefix package/src/d2q9_heat/,$(SOURCE_CU) $(HEADERS_H))
	@echo "  AUTO-DEP   $@"
	@cd package/src/d2q9_heat; $(RS) ../../$<

package/src/d2q9_heat/%:$(SRC)/%.Rt $(SRC)/d2q9_heat/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_heat -o $@ $(RTOPT) MODEL=\"d2q9_heat\" || rm $@

package/src/d2q9_heat/%:$(SRC)/d2q9_heat/%.Rt $(SRC)/d2q9_heat/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_heat -o $@ $(RTOPT) MODEL=\"d2q9_heat\" || rm $@

package/src/d2q9_heat/%:$(SRC)/d2q9_heat/%
	@echo "  CP         $@ (model)"
	@cp $< $@

package/src/d2q9_heat/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d2q9_heat_adj and destination standalone

standalone/d2q9_heat_adj/Dynamics_b.c standalone/d2q9_heat_adj/Dynamics_b.h standalone/d2q9_heat_adj/types_b.h standalone/d2q9_heat_adj/ADpre_.h standalone/d2q9_heat_adj/ADpre__b.h : standalone/d2q9_heat_adj/tapenade.run

.INTERMEDIATE : tapenade.run

standalone/d2q9_heat_adj/tapenade.run : standalone/d2q9_heat_adj standalone/d2q9_heat_adj/Dynamics.c standalone/d2q9_heat_adj/ADpre.h standalone/d2q9_heat_adj/ADpre_b.h standalone/d2q9_heat_adj/Dynamics.h standalone/d2q9_heat_adj/Node_types.h standalone/d2q9_heat_adj/types.h standalone/d2q9_heat_adj/ADset.sh
	@echo "  TAPENADE   $<"
	@(cd $<; ../../tools/makeAD)


standalone/d2q9_heat_adj/dep.mk:tools/dep.R $(addprefix standalone/d2q9_heat_adj/,$(SOURCE_CU) $(HEADERS_H))
	@echo "  AUTO-DEP   $@"
	@cd standalone/d2q9_heat_adj; $(RS) ../../$<

standalone/d2q9_heat_adj/%:$(SRC)/%.Rt $(SRC)/d2q9_heat_adj/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_heat_adj -o $@ $(RTOPT) MODEL=\"d2q9_heat_adj\" || rm $@

standalone/d2q9_heat_adj/%:$(SRC)/d2q9_heat_adj/%.Rt $(SRC)/d2q9_heat_adj/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_heat_adj -o $@ $(RTOPT) MODEL=\"d2q9_heat_adj\" || rm $@

standalone/d2q9_heat_adj/%:$(SRC)/d2q9_heat_adj/%
	@echo "  CP         $@ (model)"
	@cp $< $@

standalone/d2q9_heat_adj/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d2q9_heat_adj and destination package/src

package/src/d2q9_heat_adj/Dynamics_b.c package/src/d2q9_heat_adj/Dynamics_b.h package/src/d2q9_heat_adj/types_b.h package/src/d2q9_heat_adj/ADpre_.h package/src/d2q9_heat_adj/ADpre__b.h : package/src/d2q9_heat_adj/tapenade.run

.INTERMEDIATE : tapenade.run

package/src/d2q9_heat_adj/tapenade.run : package/src/d2q9_heat_adj package/src/d2q9_heat_adj/Dynamics.c package/src/d2q9_heat_adj/ADpre.h package/src/d2q9_heat_adj/ADpre_b.h package/src/d2q9_heat_adj/Dynamics.h package/src/d2q9_heat_adj/Node_types.h package/src/d2q9_heat_adj/types.h package/src/d2q9_heat_adj/ADset.sh
	@echo "  TAPENADE   $<"
	@(cd $<; ../../tools/makeAD)


package/src/d2q9_heat_adj/dep.mk:tools/dep.R $(addprefix package/src/d2q9_heat_adj/,$(SOURCE_CU) $(HEADERS_H))
	@echo "  AUTO-DEP   $@"
	@cd package/src/d2q9_heat_adj; $(RS) ../../$<

package/src/d2q9_heat_adj/%:$(SRC)/%.Rt $(SRC)/d2q9_heat_adj/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_heat_adj -o $@ $(RTOPT) MODEL=\"d2q9_heat_adj\" || rm $@

package/src/d2q9_heat_adj/%:$(SRC)/d2q9_heat_adj/%.Rt $(SRC)/d2q9_heat_adj/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_heat_adj -o $@ $(RTOPT) MODEL=\"d2q9_heat_adj\" || rm $@

package/src/d2q9_heat_adj/%:$(SRC)/d2q9_heat_adj/%
	@echo "  CP         $@ (model)"
	@cp $< $@

package/src/d2q9_heat_adj/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d2q9_kuper and destination standalone

standalone/d2q9_kuper/Dynamics_b.c standalone/d2q9_kuper/Dynamics_b.h standalone/d2q9_kuper/types_b.h standalone/d2q9_kuper/ADpre_.h standalone/d2q9_kuper/ADpre__b.h : standalone/d2q9_kuper/tapenade.run

.INTERMEDIATE : tapenade.run

standalone/d2q9_kuper/tapenade.run : standalone/d2q9_kuper standalone/d2q9_kuper/Dynamics.c standalone/d2q9_kuper/ADpre.h standalone/d2q9_kuper/ADpre_b.h standalone/d2q9_kuper/Dynamics.h standalone/d2q9_kuper/Node_types.h standalone/d2q9_kuper/types.h standalone/d2q9_kuper/ADset.sh
	@echo "  TAPENADE   $<"
	@(cd $<; ../../tools/makeAD)


standalone/d2q9_kuper/dep.mk:tools/dep.R $(addprefix standalone/d2q9_kuper/,$(SOURCE_CU) $(HEADERS_H))
	@echo "  AUTO-DEP   $@"
	@cd standalone/d2q9_kuper; $(RS) ../../$<

standalone/d2q9_kuper/%:$(SRC)/%.Rt $(SRC)/d2q9_kuper/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_kuper -o $@ $(RTOPT) MODEL=\"d2q9_kuper\" || rm $@

standalone/d2q9_kuper/%:$(SRC)/d2q9_kuper/%.Rt $(SRC)/d2q9_kuper/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_kuper -o $@ $(RTOPT) MODEL=\"d2q9_kuper\" || rm $@

standalone/d2q9_kuper/%:$(SRC)/d2q9_kuper/%
	@echo "  CP         $@ (model)"
	@cp $< $@

standalone/d2q9_kuper/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d2q9_kuper and destination package/src

package/src/d2q9_kuper/Dynamics_b.c package/src/d2q9_kuper/Dynamics_b.h package/src/d2q9_kuper/types_b.h package/src/d2q9_kuper/ADpre_.h package/src/d2q9_kuper/ADpre__b.h : package/src/d2q9_kuper/tapenade.run

.INTERMEDIATE : tapenade.run

package/src/d2q9_kuper/tapenade.run : package/src/d2q9_kuper package/src/d2q9_kuper/Dynamics.c package/src/d2q9_kuper/ADpre.h package/src/d2q9_kuper/ADpre_b.h package/src/d2q9_kuper/Dynamics.h package/src/d2q9_kuper/Node_types.h package/src/d2q9_kuper/types.h package/src/d2q9_kuper/ADset.sh
	@echo "  TAPENADE   $<"
	@(cd $<; ../../tools/makeAD)


package/src/d2q9_kuper/dep.mk:tools/dep.R $(addprefix package/src/d2q9_kuper/,$(SOURCE_CU) $(HEADERS_H))
	@echo "  AUTO-DEP   $@"
	@cd package/src/d2q9_kuper; $(RS) ../../$<

package/src/d2q9_kuper/%:$(SRC)/%.Rt $(SRC)/d2q9_kuper/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_kuper -o $@ $(RTOPT) MODEL=\"d2q9_kuper\" || rm $@

package/src/d2q9_kuper/%:$(SRC)/d2q9_kuper/%.Rt $(SRC)/d2q9_kuper/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d2q9_kuper -o $@ $(RTOPT) MODEL=\"d2q9_kuper\" || rm $@

package/src/d2q9_kuper/%:$(SRC)/d2q9_kuper/%
	@echo "  CP         $@ (model)"
	@cp $< $@

package/src/d2q9_kuper/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d3q19 and destination standalone

standalone/d3q19/Dynamics_b.c standalone/d3q19/Dynamics_b.h standalone/d3q19/types_b.h standalone/d3q19/ADpre_.h standalone/d3q19/ADpre__b.h : standalone/d3q19/tapenade.run

.INTERMEDIATE : tapenade.run

standalone/d3q19/tapenade.run : standalone/d3q19 standalone/d3q19/Dynamics.c standalone/d3q19/ADpre.h standalone/d3q19/ADpre_b.h standalone/d3q19/Dynamics.h standalone/d3q19/Node_types.h standalone/d3q19/types.h standalone/d3q19/ADset.sh
	@echo "  TAPENADE   $<"
	@(cd $<; ../../tools/makeAD)


standalone/d3q19/dep.mk:tools/dep.R $(addprefix standalone/d3q19/,$(SOURCE_CU) $(HEADERS_H))
	@echo "  AUTO-DEP   $@"
	@cd standalone/d3q19; $(RS) ../../$<

standalone/d3q19/%:$(SRC)/%.Rt $(SRC)/d3q19/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d3q19 -o $@ $(RTOPT) MODEL=\"d3q19\" || rm $@

standalone/d3q19/%:$(SRC)/d3q19/%.Rt $(SRC)/d3q19/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d3q19 -o $@ $(RTOPT) MODEL=\"d3q19\" || rm $@

standalone/d3q19/%:$(SRC)/d3q19/%
	@echo "  CP         $@ (model)"
	@cp $< $@

standalone/d3q19/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d3q19 and destination package/src

package/src/d3q19/Dynamics_b.c package/src/d3q19/Dynamics_b.h package/src/d3q19/types_b.h package/src/d3q19/ADpre_.h package/src/d3q19/ADpre__b.h : package/src/d3q19/tapenade.run

.INTERMEDIATE : tapenade.run

package/src/d3q19/tapenade.run : package/src/d3q19 package/src/d3q19/Dynamics.c package/src/d3q19/ADpre.h package/src/d3q19/ADpre_b.h package/src/d3q19/Dynamics.h package/src/d3q19/Node_types.h package/src/d3q19/types.h package/src/d3q19/ADset.sh
	@echo "  TAPENADE   $<"
	@(cd $<; ../../tools/makeAD)


package/src/d3q19/dep.mk:tools/dep.R $(addprefix package/src/d3q19/,$(SOURCE_CU) $(HEADERS_H))
	@echo "  AUTO-DEP   $@"
	@cd package/src/d3q19; $(RS) ../../$<

package/src/d3q19/%:$(SRC)/%.Rt $(SRC)/d3q19/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d3q19 -o $@ $(RTOPT) MODEL=\"d3q19\" || rm $@

package/src/d3q19/%:$(SRC)/d3q19/%.Rt $(SRC)/d3q19/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d3q19 -o $@ $(RTOPT) MODEL=\"d3q19\" || rm $@

package/src/d3q19/%:$(SRC)/d3q19/%
	@echo "  CP         $@ (model)"
	@cp $< $@

package/src/d3q19/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d3q19_adj and destination standalone

standalone/d3q19_adj/Dynamics_b.c standalone/d3q19_adj/Dynamics_b.h standalone/d3q19_adj/types_b.h standalone/d3q19_adj/ADpre_.h standalone/d3q19_adj/ADpre__b.h : standalone/d3q19_adj/tapenade.run

.INTERMEDIATE : tapenade.run

standalone/d3q19_adj/tapenade.run : standalone/d3q19_adj standalone/d3q19_adj/Dynamics.c standalone/d3q19_adj/ADpre.h standalone/d3q19_adj/ADpre_b.h standalone/d3q19_adj/Dynamics.h standalone/d3q19_adj/Node_types.h standalone/d3q19_adj/types.h standalone/d3q19_adj/ADset.sh
	@echo "  TAPENADE   $<"
	@(cd $<; ../../tools/makeAD)


standalone/d3q19_adj/dep.mk:tools/dep.R $(addprefix standalone/d3q19_adj/,$(SOURCE_CU) $(HEADERS_H))
	@echo "  AUTO-DEP   $@"
	@cd standalone/d3q19_adj; $(RS) ../../$<

standalone/d3q19_adj/%:$(SRC)/%.Rt $(SRC)/d3q19_adj/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d3q19_adj -o $@ $(RTOPT) MODEL=\"d3q19_adj\" || rm $@

standalone/d3q19_adj/%:$(SRC)/d3q19_adj/%.Rt $(SRC)/d3q19_adj/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d3q19_adj -o $@ $(RTOPT) MODEL=\"d3q19_adj\" || rm $@

standalone/d3q19_adj/%:$(SRC)/d3q19_adj/%
	@echo "  CP         $@ (model)"
	@cp $< $@

standalone/d3q19_adj/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d3q19_adj and destination package/src

package/src/d3q19_adj/Dynamics_b.c package/src/d3q19_adj/Dynamics_b.h package/src/d3q19_adj/types_b.h package/src/d3q19_adj/ADpre_.h package/src/d3q19_adj/ADpre__b.h : package/src/d3q19_adj/tapenade.run

.INTERMEDIATE : tapenade.run

package/src/d3q19_adj/tapenade.run : package/src/d3q19_adj package/src/d3q19_adj/Dynamics.c package/src/d3q19_adj/ADpre.h package/src/d3q19_adj/ADpre_b.h package/src/d3q19_adj/Dynamics.h package/src/d3q19_adj/Node_types.h package/src/d3q19_adj/types.h package/src/d3q19_adj/ADset.sh
	@echo "  TAPENADE   $<"
	@(cd $<; ../../tools/makeAD)


package/src/d3q19_adj/dep.mk:tools/dep.R $(addprefix package/src/d3q19_adj/,$(SOURCE_CU) $(HEADERS_H))
	@echo "  AUTO-DEP   $@"
	@cd package/src/d3q19_adj; $(RS) ../../$<

package/src/d3q19_adj/%:$(SRC)/%.Rt $(SRC)/d3q19_adj/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d3q19_adj -o $@ $(RTOPT) MODEL=\"d3q19_adj\" || rm $@

package/src/d3q19_adj/%:$(SRC)/d3q19_adj/%.Rt $(SRC)/d3q19_adj/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d3q19_adj -o $@ $(RTOPT) MODEL=\"d3q19_adj\" || rm $@

package/src/d3q19_adj/%:$(SRC)/d3q19_adj/%
	@echo "  CP         $@ (model)"
	@cp $< $@

package/src/d3q19_adj/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d3q19_heat and destination standalone

standalone/d3q19_heat/Dynamics_b.c standalone/d3q19_heat/Dynamics_b.h standalone/d3q19_heat/types_b.h standalone/d3q19_heat/ADpre_.h standalone/d3q19_heat/ADpre__b.h : standalone/d3q19_heat/tapenade.run

.INTERMEDIATE : tapenade.run

standalone/d3q19_heat/tapenade.run : standalone/d3q19_heat standalone/d3q19_heat/Dynamics.c standalone/d3q19_heat/ADpre.h standalone/d3q19_heat/ADpre_b.h standalone/d3q19_heat/Dynamics.h standalone/d3q19_heat/Node_types.h standalone/d3q19_heat/types.h standalone/d3q19_heat/ADset.sh
	@echo "  TAPENADE   $<"
	@(cd $<; ../../tools/makeAD)


standalone/d3q19_heat/dep.mk:tools/dep.R $(addprefix standalone/d3q19_heat/,$(SOURCE_CU) $(HEADERS_H))
	@echo "  AUTO-DEP   $@"
	@cd standalone/d3q19_heat; $(RS) ../../$<

standalone/d3q19_heat/%:$(SRC)/%.Rt $(SRC)/d3q19_heat/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d3q19_heat -o $@ $(RTOPT) MODEL=\"d3q19_heat\" || rm $@

standalone/d3q19_heat/%:$(SRC)/d3q19_heat/%.Rt $(SRC)/d3q19_heat/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d3q19_heat -o $@ $(RTOPT) MODEL=\"d3q19_heat\" || rm $@

standalone/d3q19_heat/%:$(SRC)/d3q19_heat/%
	@echo "  CP         $@ (model)"
	@cp $< $@

standalone/d3q19_heat/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d3q19_heat and destination package/src

package/src/d3q19_heat/Dynamics_b.c package/src/d3q19_heat/Dynamics_b.h package/src/d3q19_heat/types_b.h package/src/d3q19_heat/ADpre_.h package/src/d3q19_heat/ADpre__b.h : package/src/d3q19_heat/tapenade.run

.INTERMEDIATE : tapenade.run

package/src/d3q19_heat/tapenade.run : package/src/d3q19_heat package/src/d3q19_heat/Dynamics.c package/src/d3q19_heat/ADpre.h package/src/d3q19_heat/ADpre_b.h package/src/d3q19_heat/Dynamics.h package/src/d3q19_heat/Node_types.h package/src/d3q19_heat/types.h package/src/d3q19_heat/ADset.sh
	@echo "  TAPENADE   $<"
	@(cd $<; ../../tools/makeAD)


package/src/d3q19_heat/dep.mk:tools/dep.R $(addprefix package/src/d3q19_heat/,$(SOURCE_CU) $(HEADERS_H))
	@echo "  AUTO-DEP   $@"
	@cd package/src/d3q19_heat; $(RS) ../../$<

package/src/d3q19_heat/%:$(SRC)/%.Rt $(SRC)/d3q19_heat/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d3q19_heat -o $@ $(RTOPT) MODEL=\"d3q19_heat\" || rm $@

package/src/d3q19_heat/%:$(SRC)/d3q19_heat/%.Rt $(SRC)/d3q19_heat/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d3q19_heat -o $@ $(RTOPT) MODEL=\"d3q19_heat\" || rm $@

package/src/d3q19_heat/%:$(SRC)/d3q19_heat/%
	@echo "  CP         $@ (model)"
	@cp $< $@

package/src/d3q19_heat/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d3q19_heat_adj and destination standalone

standalone/d3q19_heat_adj/Dynamics_b.c standalone/d3q19_heat_adj/Dynamics_b.h standalone/d3q19_heat_adj/types_b.h standalone/d3q19_heat_adj/ADpre_.h standalone/d3q19_heat_adj/ADpre__b.h : standalone/d3q19_heat_adj/tapenade.run

.INTERMEDIATE : tapenade.run

standalone/d3q19_heat_adj/tapenade.run : standalone/d3q19_heat_adj standalone/d3q19_heat_adj/Dynamics.c standalone/d3q19_heat_adj/ADpre.h standalone/d3q19_heat_adj/ADpre_b.h standalone/d3q19_heat_adj/Dynamics.h standalone/d3q19_heat_adj/Node_types.h standalone/d3q19_heat_adj/types.h standalone/d3q19_heat_adj/ADset.sh
	@echo "  TAPENADE   $<"
	@(cd $<; ../../tools/makeAD)


standalone/d3q19_heat_adj/dep.mk:tools/dep.R $(addprefix standalone/d3q19_heat_adj/,$(SOURCE_CU) $(HEADERS_H))
	@echo "  AUTO-DEP   $@"
	@cd standalone/d3q19_heat_adj; $(RS) ../../$<

standalone/d3q19_heat_adj/%:$(SRC)/%.Rt $(SRC)/d3q19_heat_adj/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d3q19_heat_adj -o $@ $(RTOPT) MODEL=\"d3q19_heat_adj\" || rm $@

standalone/d3q19_heat_adj/%:$(SRC)/d3q19_heat_adj/%.Rt $(SRC)/d3q19_heat_adj/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d3q19_heat_adj -o $@ $(RTOPT) MODEL=\"d3q19_heat_adj\" || rm $@

standalone/d3q19_heat_adj/%:$(SRC)/d3q19_heat_adj/%
	@echo "  CP         $@ (model)"
	@cp $< $@

standalone/d3q19_heat_adj/%:$(SRC)/%
	@echo "  CP         $@"
	@cp $< $@

# for model d3q19_heat_adj and destination package/src

package/src/d3q19_heat_adj/Dynamics_b.c package/src/d3q19_heat_adj/Dynamics_b.h package/src/d3q19_heat_adj/types_b.h package/src/d3q19_heat_adj/ADpre_.h package/src/d3q19_heat_adj/ADpre__b.h : package/src/d3q19_heat_adj/tapenade.run

.INTERMEDIATE : tapenade.run

package/src/d3q19_heat_adj/tapenade.run : package/src/d3q19_heat_adj package/src/d3q19_heat_adj/Dynamics.c package/src/d3q19_heat_adj/ADpre.h package/src/d3q19_heat_adj/ADpre_b.h package/src/d3q19_heat_adj/Dynamics.h package/src/d3q19_heat_adj/Node_types.h package/src/d3q19_heat_adj/types.h package/src/d3q19_heat_adj/ADset.sh
	@echo "  TAPENADE   $<"
	@(cd $<; ../../tools/makeAD)


package/src/d3q19_heat_adj/dep.mk:tools/dep.R $(addprefix package/src/d3q19_heat_adj/,$(SOURCE_CU) $(HEADERS_H))
	@echo "  AUTO-DEP   $@"
	@cd package/src/d3q19_heat_adj; $(RS) ../../$<

package/src/d3q19_heat_adj/%:$(SRC)/%.Rt $(SRC)/d3q19_heat_adj/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d3q19_heat_adj -o $@ $(RTOPT) MODEL=\"d3q19_heat_adj\" || rm $@

package/src/d3q19_heat_adj/%:$(SRC)/d3q19_heat_adj/%.Rt $(SRC)/d3q19_heat_adj/Dynamics.R $(SRC)/conf.R
	@echo "  RT         $@ (model)"
	@$(RT) -q -f $< -I $(SRC),$(SRC)/d3q19_heat_adj -o $@ $(RTOPT) MODEL=\"d3q19_heat_adj\" || rm $@

package/src/d3q19_heat_adj/%:$(SRC)/d3q19_heat_adj/%
	@echo "  CP         $@ (model)"
	@cp $< $@

package/src/d3q19_heat_adj/%:$(SRC)/%
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

