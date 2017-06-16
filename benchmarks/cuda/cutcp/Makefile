include ../../common/make.config

# C compiler
CC = gcc
CC_FLAGS = -g  -O2

# CUDA compiler
NVCC = $(CUDA_DIR)/bin/nvcc
NVCC_FLAGS = -I$(CUDA_DIR)/include

# 'make dbg=1' enables NVCC debugging
ifeq ($(dbg),1)
	NVCC_FLAGS += -g -O0
else
	NVCC_FLAGS += -O2
endif

# 'make emu=1' compiles the CUDA kernels for emulation
ifeq ($(emu),1)
	NVCC_FLAGS += -deviceemu
endif


cutcp: readatom.o output.o excl.o cutoff6overlap.o cutcpu.o
	$(NVCC) $(NVCC_FLAGS) main.cu *.o -o cutcp -L$(CUDA_LIB_DIR) -lcudart -lm -lstdc++

cutoff6overlap.o: cutoff6overlap.cu cutoff.h readatom.o output.o cutcpu.o excl.o
	$(NVCC) $(NVCC_FLAGS) cutoff6overlap.cu *.o -c

readatom.o: readatom.c
	$(CC) $(CC_FLAGS) readatom.c -c

output.o: output.c output.h
	$(CC) $(CC_FLAGS) output.c -c

excl.o: excl.c
	$(CC) $(CC_FLAGS) excl.c -c

cutcpu.o: cutcpu.c
	$(CC) $(CC_FLAGS) cutcpu.c -c


clean:
	rm -f *.o *~ bfs
