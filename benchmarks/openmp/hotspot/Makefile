# C compiler
CC = icpc
CC_FLAGS = -g -qopenmp -O2 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lrt -lpthread

all: hotspot 


hotspot: hotspot_openmp.cpp Makefile 
	$(CC) $(CC_FLAGS) hotspot_openmp.cpp -o hotspot 


clean:
	rm -f hotspot *.out *.o

test:
	./hotspot 64 64 100 4 ./temp_64 ./power_64 test.out
