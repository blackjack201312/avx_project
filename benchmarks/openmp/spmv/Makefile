# C Compiler
CC = icc
CC_FLAGS = -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lrt -lpthread -qopenmp
CC_LINK = -I../../common/include
APP=$(shell basename $(CURDIR))
OBJS = main.o file.o mmio.o convert_dataset.o

$(APP): $(OBJS) ../../common/src/parboil.c
	$(CC) $(CC_FLAGS) $(CC_LINK) $(OBJS) ../../common/src/parboil.c -o $(APP)

main.o: main.c ../../common/include/parboil.h
	$(CC) $(CC_FLAGS) $(CC_LINK) -c main.c

file.o: file.c file.h
	$(CC) $(CC_FLAGS) $(CC_LINK) -c file.c

mmio.o: mmio.c mmio.h
	$(CC) -c mmio.c -o mmio.o
	
convert_dataset.o: convert_dataset.c
	$(CC) -c convert_dataset.c


clean:
	rm -rf *.o $(APP)

test: test.dat test.bin
	./$(APP) -i test.dat,test.bin
