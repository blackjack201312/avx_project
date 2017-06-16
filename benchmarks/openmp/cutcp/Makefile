# C Compiler
CC = icc
CC_FLAGS = -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lrt -lpthread -qopenmp -lstdc++ -lgomp

CC_LINK = -I../../common/include -I./
APP = $(shell basename $(CURDIR))
OBJS =  main.o readatom.o cutcpu.o excl.o output.o

$(APP): ../../common/src/parboil.c $(OBJS)
	$(CC) $(CC_FLAGS) $(CC_LINK) $(OBJS) ../../common/src/parboil.c -o $(APP)

main.o: main.c readatom.o cutcpu.o excl.o output.o ../../common/include/parboil.h
	$(CC) $(CC_FLAGS) $(CC_LINK) -c main.c

readatom.o: readatom.c atom.h
	$(CC) $(CC_LINK) $(CC_FLAGS) -c readatom.c

cutcpu.o: cutcpu.c cutoff.h
	$(CC) $(CC_LINK) $(CC_FLAGS) -c cutcpu.c

excl.o: atom.h cutoff.h excl.c
	$(CC) $(CC_LINK) $(CC_FLAGS) -c excl.c

output.o: atom.h cutoff.h output.c
	$(CC) $(CC_LINK) $(CC_FLAGS) -c output.c

clean:
	rm -rf *.o $(APP)

test: test.dat
	./$(APP) -i test.dat
