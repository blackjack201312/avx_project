include ../../common/make.config

# C compiler
CC = g++
CC_FLAGS = -g -O2 

histo: util.o OpenCL_common.o
	$(CC) $(CC_FLAGS) main.cpp OpenCL_common.o util.o -o histo -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL

# %.o: %.[ch]
# 	$(CC) $(CC_FLAGS) $< -c
	
OpenCL_common.o: OpenCL_common.cpp 
	$(CC) $(CC_FLAGS) OpenCL_common.cpp -c -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL
	
util.o: 
	$(CC) $(CC_FLAGS) util.cpp -c
clean:
	rm -f *.o *~ histo *.linkinfo
