LOCAL_CC = gcc -g -O3 -Wall
CC = icc
CFLAGS = -qopenmp -Wall -O3 -fp-model fast=2 -fma -qoffload-option,mic,compiler,"-qno-opt-prefetch"


all : nn

clean :
	rm -rf *.o nn

nn : nn_openmp.c
	$(CC) -o $@ $< $(LDFLAGS) $(CFLAGS) 

hurricane_gen : hurricane_gen.c
	$(LOCAL_CC) -o $@ $< -lm

