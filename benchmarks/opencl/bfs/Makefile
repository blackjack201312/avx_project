include ../../common/make.config

# C compiler
CC = g++
CC_FLAGS = -g -O2 

bfs: OpenCL_common.o
	$(CC) $(CC_FLAGS) main.cpp OpenCL_common.o -o bfs -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL

# %.o: %.[ch]
# 	$(CC) $(CC_FLAGS) $< -c
	
OpenCL_common.o: OpenCL_common.cpp 
	$(CC) $(CC_FLAGS) OpenCL_common.cpp -c -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL
	
clean:
	rm -f *.o *~ bfs *.linkinfo
