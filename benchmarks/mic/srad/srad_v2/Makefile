# C compiler
CC = icc
CC_FLAGS = -Wall -O3 -qopenmp -fp-model fast=2 -fma -qoffload-option,mic,compiler,"-qno-opt-prefetch"

bfs: 
	$(CC) $(CC_FLAGS) srad.cpp -o srad 

clean:
	rm -f srad
