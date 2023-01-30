clean:
	rm a.out

all: a.out

a.out: mainGeneral.cu
	nvcc mainGeneral.cu	

send:
	git commit -a -m "automatic commit"
	git push
