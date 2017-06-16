include ../../common/make.config

# C compiler
CC = g++
CC_FLAGS = -g -O2 

stencil: file.o
	$(CC) $(CC_FLAGS) main.c file.o -o stencil -I$(OPENCL_INC) -L$(OPENCL_LIB) -lm

# %.o: %.[ch]
# 	$(CC) $(CC_FLAGS) $< -c
	
file.o: file.cc 
	$(CC) $(CC_FLAGS) file.cc -c
	
clean:
	rm -f *.o *~ bfs *.linkinfo
