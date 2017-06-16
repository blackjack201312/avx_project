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


bfs: main.cu kernel.cu config.h
	$(NVCC) $(NVCC_FLAGS) main.cu -o bfs -L$(CUDA_LIB_DIR) -lcudart -lm -lstdc++


clean:
	rm -f *.o *~ bfs
