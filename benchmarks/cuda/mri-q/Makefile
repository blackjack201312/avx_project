# (c) 2007 The Board of Trustees of the University of Illinois.

LANGUAGE=cuda

SRCDIR_OBJS=file.o main.o
APP_LDFLAGS=-lm -lstdc++
APP_CUDACFLAGS=--use_fast_math

CC=gcc
NVCC=nvcc

mri-q: main.cu computeQ.cu file.o
	$(NVCC) main.cu file.o -o mri-q $(APP_CUDACFLAGS) $(APP_LDFLAGS)

file.o: file.cc file.h
	$(CC) file.cc -c $(APP_LDFLAGS)

clean:
	rm *.o mri-q
