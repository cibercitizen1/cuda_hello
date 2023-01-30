
include ../makefile_h

a.out: mainGeneral.cu
	git pull
	rm a.out || true
	nvcc mainGeneral.cu	
	./a.out
