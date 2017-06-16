LOCAL_CC = icc -g -qopenmp -O2 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lrt -lpthread
CC = icc
CFLAGS = -lm -Wall -g -qopenmp -O2 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lrt -lpthread


all : nn

clean :
	rm -rf *.o nn

nn : nn_openmp.c
	$(CC) -o $@ $< $(LDFLAGS) $(CFLAGS) 

hurricane_gen : hurricane_gen.c
	$(LOCAL_CC) -o $@ $< -lm

test:
	./nn testlist 10 30 90
