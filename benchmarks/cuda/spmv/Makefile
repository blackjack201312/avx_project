# (c) 2007 The Board of Trustees of the University of Illinois.

LANGUAGE=cuda
TOOLS_SRC=../../common/convert-dataset
SRCDIR_OBJS=main.o file.o gpu_info.o 
APP_LDFLAGS= -lm -lstdc++
APP_CUDACFLAGS=-I$(TOOLS_SRC)
APP_CFLAGS=-I$(TOOLS_SRC)

# To use slow (but precise) math routines
ifeq ($(slowmath), 1)
	EXTRA_CFLAGS += -fp-model precise
else
	EXTRA_CFLAGS += -ffast-math
	EXTRA_CXXFLAGS += -ffast-math
	EXTRA_CUDACFLAGS += --use_fast_math
endif

# To use the MKL library
ifeq ($(mkl), 1)
	EXTRA_LIBS += -L/opt/intel/mkl/9.1.023/lib/32 \
		-lmkl -lvml -lguide -lpthread
	INCLUDES += -I/opt/intel/mkl/9.1.023/include
	EXTRA_CFLAGS += -DMKL
endif

include $(TOOLS_SRC)/commontools.mk

CC=gcc
NVCC=nvcc

spmv: main.cu file.o gpu_info.o
	$(NVCC) main.cu *.o $(TOOLS_SRC)/*.o -o spmv $(APP_CUDACFLAGS) -L$(TOOLS_SRC)

file.o: file.cc file.h
	$(CC) file.cc -c $(APP_CFLAGS)

gpu_info.o: gpu_info.cc gpu_info.h
	$(CC) gpu_info.cc -c $(APP_CFLAGS)

clean:
	rm -f *.o *~ spmv


