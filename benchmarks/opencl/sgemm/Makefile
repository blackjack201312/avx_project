include ../../common/make.config

# C compiler
CC = g++
CC_FLAGS = -g -O2 -ffast-math -g3

sgemm: io.o
	$(CC) $(CC_FLAGS) main.cc io.o -o sgemm -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL

# %.o: %.[ch]
# 	$(CC) $(CC_FLAGS) $< -c
	
io.o: io.cc 
	$(CC) $(CC_FLAGS) io.cc -c -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL
	
clean:
	rm -f *.o *~ sgemm *.linkinfo
