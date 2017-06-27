# C Compiler
CC = icpc
CC_FLAGS = -O3 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lrt -lpthread -qopenmp
CC_LINK = -I../../common/include
OBJS = io.o sgemm_kernel.o main.o 

APP=$(shell basename $(CURDIR))

$(APP): $(OBJS) ../../common/src/parboil.c
	$(CC) $(CC_FLAGS) $(CC_LINK) $(OBJS) ../../common/src/parboil.c -o $(APP)

main.o: main.cc ../../common/include/parboil.h
	$(CC) $(CC_FLAGS) $(CC_LINK) -c main.cc

io.o: io.cc
	$(CC) $(CC_FLAGS) $(CC_LINK) -c io.cc

sgemm_kernel.o: sgemm_kernel_simd.cc
	$(CC) $(CC_FLAGS) $(CC_LINK) -c sgemm_kernel.cc

clean:
	rm -rf *.o $(APP)

test: matrix1.txt matrix2.txt matrix2t.txt
	./$(APP) -i matrix1.txt,matrix2.txt,matrix2t.txt
