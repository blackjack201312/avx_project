# C Compiler
CC = icc
CC_FLAGS = -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lrt -lpthread -qopenmp
CC_LINK = -I../../common/include
APP=$(shell basename $(CURDIR))
OBJS = main.o file.o kernels.o

$(APP): $(OBJS) ../../common/src/parboil.c
	$(CC) $(CC_FLAGS) $(CC_LINK) $(OBJS) ../../common/src/parboil.c -o $(APP)

main.o: main.c ../../common/include/parboil.h
	$(CC) $(CC_FLAGS) $(CC_LINK) -c main.c

kernels.o: kernels.c common.h
	$(CC) $(CC_FLAGS) $(CC_LINK) -c kernels.c

file.o: file.c file.h
	$(CC) $(CC_FLAGS) $(CC_LINK) -c file.c


clean:
	rm -rf *.o $(APP)

test: test.dat
	./$(APP) -i test.dat 128 128 32 100
