
include ../makefile_h

UTIL_DIR			= ../80_util
IMAGE_DIR			= ../02_lib_images
INCLUDE_FLAG	= -I$(UTIL_DIR) -I$(IMAGE_DIR) 
DIR_LIBS_FLAG	= -L$(UTIL_DIR) -L$(IMAGE_DIR)
LIBS			= -lmyutil -limages
CUDA_LIBS			= -lmyutilCuda -limagesCuda

compile: mainGeneral.cu
	make -C $(UTIL_DIR) cudalib
	make -C $(IMAGE_DIR) cudalib
	nvcc $(INCLUDE_FLAG) mainGeneral.cu	$(DIR_LIBS_FLAG) $(CUDA_LIBS)

alinna: 
	#git pull

run: compile
	a.out image=../90_data/patoC.bmp 

cpimage:
	scp bataller@alinna.dsic.upv.es:/tmp/corrected.bmp .
	
clean:
	rm -f a.out
	rm -f corrected.bmp

