include ../../common/make.config

# C compiler
TOOLS_SRC=../../../tools/convert-dataset
CC = gcc
CC_FLAGS = -g -O2 -I$(TOOLS_SRC) -L$(TOOLS_SRC)

spmv: ocl.o file.o gpu_info.o 
	$(CC) $(CC_FLAGS) main.c ocl.o file.o gpu_info.o convert_dataset.o -o bfs -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL -lm

# %.o: %.[ch]
# 	$(CC) $(CC_FLAGS) $< -c
	
ocl.o: ocl.c
	$(CC) $(CC_FLAGS) ocl.c -c -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL
	
file.o: file.c 
	$(CC) $(CC_FLAGS) file.c -c # -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL

gpu_info.o: gpu_info.c
	$(CC) $(CC_FLAGS) gpu_info.c -c # -I$(OPENCL_INC) -L$(OPENCL_LIB) -lOpenCL
clean:
	rm -f *.o *~ spmv *.linkinfo
