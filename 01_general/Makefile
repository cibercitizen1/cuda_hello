clean:
	rm a.out

all: a.out

a.out: mainGeneral.cu
	rm a.out || true
	git pull
	nvcc mainGeneral.cu	
	./a.out

send:
	git commit -a -m "automatic commit"
	git push
