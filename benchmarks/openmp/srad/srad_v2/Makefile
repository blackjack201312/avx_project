# C compiler
CC = icpc
CC_FLAGS = -g -qopenmp -O2 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lrt -lpthread

bfs: 
	$(CC) $(CC_FLAGS) srad.cpp -o srad 

clean:
	rm -f srad
