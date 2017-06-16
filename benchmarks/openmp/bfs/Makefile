# C Compiler
CC = icpc
CC_FLAGS = -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lrt -lpthread -qopenmp
CC_LINK = -I../../common/include
APP=$(shell basename $(CURDIR))
OBJS = main.o

$(APP): $(OBJS) ../../common/src/parboil.c
	$(CC) $(CC_FLAGS) $(CC_LINK) $(OBJS) ../../common/src/parboil.c -o $(APP)

main.o: main.cc ../../common/include/parboil.h
	$(CC) $(CC_FLAGS) $(CC_LINK) -c main.cc

clean:
	rm -rf *.o $(APP)

test: test.dat
	./$(APP) -i test.dat
