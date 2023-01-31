
include ../makefile_h

LIB_DIR			= ../80_util
INCLUDE_FLAG	= -I$(LIB_DIR) -I.
DIR_LIBS_FLAG	= -L$(LIB_DIR)
LIBS			= -lmyutil

compile: mainGeneral.cu
	make -C $(LIB_DIR) cudalib
	nvcc $(INCLUDE_FLAG) mainGeneral.cu	$(DIR_LIBS_FLAG) $(LIBS)

a.out: mainGeneral.cu
	git pull
	make -C $(LIB_DIR) cudalib
	nvcc $(INCLUDE_FLAG) mainGeneral.cu	$(DIR_LIBS_FLAG) $(LIBS)
	./a.out
