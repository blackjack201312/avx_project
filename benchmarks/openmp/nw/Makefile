# C compiler
CC = icpc
ICC = icc
CC_FLAGS = -qopenmp -O2 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lrt -lpthread

all: needle

needle: 
	$(CC) $(CC_FLAGS) needle.cpp -o needle 

clean:
	rm -f needle

test:
	./needle 2048 10 2
