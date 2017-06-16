# C Compiler
CC = icc
CC_FLAGS = -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lrt -lpthread -qopenmp
CC_LINK = -I../../common/include
APP = $(shell basename $(CURDIR))
OBJS = main.o util.o  

$(APP): $(OBJS) ../../common/src/parboil.c
	$(CC) $(CC_FLAGS) $(CC_LINK) $(OBJS) ../../common/src/parboil.c -o $(APP)

main.o: util.h main.c ../../common/include/parboil.h
	$(CC) $(CC_FLAGS) $(CC_LINK) -c main.c

util.o: bmp.h util.h util.c
	$(CC) $(CC_FLAGS) $(CC_LINK) -c util.c 

clean:
	rm -rf *.o $(APP)

test: test.dat
	./$(APP) 1 -i test.dat
