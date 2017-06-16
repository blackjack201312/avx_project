# (c) 2007 The Board of Trustees of the University of Illinois.

LANGUAGE=cuda
SRCDIR_OBJS=main.o histo_final.o histo_intermediates.o histo_main.o histo_prescan.o util.o
APP_CUDACFLAGS=-gencode arch=compute_30,code=sm_30 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_61,code=sm_61

NVCC=nvcc

histo: histo_final.o histo_intermediates.o histo_main.o histo_prescan.o util.o
	$(NVCC) main.cu *.o -o histo $(APP_CUDACFLAGS)

histo_final.o: histo_final.cu
	$(NVCC) histo_final.cu -c $(APP_CUDACFLAGS)

histo_intermediates.o: histo_intermediates.cu
	$(NVCC) histo_intermediates.cu -c $(APP_CUDACFLAGS)

histo_main.o: histo_main.cu
	$(NVCC) histo_main.cu -c $(APP_CUDACFLAGS)

histo_prescan.o: histo_prescan.cu
	$(NVCC) histo_prescan.cu -c $(APP_CUDACFLAGS)

util.o: util.cu
	$(NVCC) util.cu -c $(APP_CUDACFLAGS)


