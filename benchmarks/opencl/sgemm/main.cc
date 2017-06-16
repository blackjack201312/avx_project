/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/* 
 * Main entry of dense matrix-matrix multiplication kernel
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <malloc.h>
#include <vector>
#include <iostream>
#include <CL/cl.h>

// I/O routines
extern bool readColMajorMatrixFile(const char *fn, int &nr_row, int &nr_col, std::vector<float>&v);
extern bool writeColMajorMatrixFile(const char *fn, int, int, std::vector<float>&);
extern char* readFile(const char*);

// Parameters of tile sizes
#define TILE_SZ 16

#define CHECK_ERROR(errorMessage)           \
  if(clStatus != CL_SUCCESS)                \
  {                                         \
     std::cout<<errorMessage<<" Error!\n";  \
     std::cout<<"Line: "<<__LINE__<<"\n";   \
     exit(1);                               \
  }

double gettime()
{
	struct timeval t;
	gettimeofday(&t, NULL);
	return t.tv_sec + t.tv_usec * 1e-6;
}


void basicSgemm( char transa, char transb, int m, int n, int k, float alpha, cl_mem A, int lda, cl_mem B, int ldb, float beta, cl_mem C, int ldc, cl_kernel clKernel, cl_command_queue clCommandQueue )
{
  if ((transa != 'N') && (transa != 'n')) {
    std::cerr << "unsupported value of 'transa' in regtileSgemm()" << std::endl;
    return;
  }
  
  if ((transb != 'T') && (transb != 't')) {
    std::cerr << "unsupported value of 'transb' in regtileSgemm()" << std::endl;
    return;
  }
  
  // In this code we assume the matrix sizes are multiple of tile size
  if ((m%TILE_SZ) || (n%TILE_SZ)) {
    std::cerr << "unsupported size of matrix. m should be multiple of " << TILE_SZ
      << "; n should be multiple of " << TILE_SZ << std::endl;
  }

  size_t db[2] = {TILE_SZ,TILE_SZ};
  size_t dg[2] = {m/TILE_SZ*db[0],n/TILE_SZ*db[1]};

  cl_int clStatus;
 
  clStatus = clSetKernelArg(clKernel,0,sizeof(cl_mem),(void*)&A);
  clStatus = clSetKernelArg(clKernel,1,sizeof(int),(void*)&lda);
  clStatus = clSetKernelArg(clKernel,2,sizeof(cl_mem),(void*)&B);
  clStatus = clSetKernelArg(clKernel,3,sizeof(int),(void*)&ldb);
  clStatus = clSetKernelArg(clKernel,4,sizeof(cl_mem),(void*)&C);
  clStatus = clSetKernelArg(clKernel,5,sizeof(int),(void*)&ldc);
  clStatus = clSetKernelArg(clKernel,6,sizeof(int),(void*)&k);
  clStatus = clSetKernelArg(clKernel,7,sizeof(float),(void*)&alpha);
  clStatus = clSetKernelArg(clKernel,8,sizeof(float),(void*)&beta);
  CHECK_ERROR("clSetKernelArg")

  const char* env_itrs = getenv("ITERS");
  int nIter = (env_itrs != NULL) ? atoi(env_itrs) : 1;
  const char* env_secs = getenv("SECS");
  int secs = (env_secs != NULL) ? atoi(env_secs) : 10;
  int timeRestrict = (env_secs != NULL) ? 1 : 0;

  double start_t;
  double end_t;
  double total_s = 0;
  int c = 0;

  for(int i = -20; i < nIter;i++){
	  start_t = gettime();

  clStatus = clEnqueueNDRangeKernel(clCommandQueue,clKernel,2,NULL,dg,db,0,NULL,NULL);
  CHECK_ERROR("clEnqueueNDRangeKernel")

  clStatus = clFinish(clCommandQueue); 
  CHECK_ERROR("clFinish")

	  end_t = gettime();
  
  	if(i == -1)
	{
		if(timeRestrict)
		{
			double tPerIter = total_s / c;
			printf("Estimate %lf s.\n", tPerIter);
			nIter = int((double)secs / tPerIter) + 1;
			printf("Adjust %d iterations to meet %d seconds.\n", nIter, secs);
		}
		total_s = 0;
	}
	else
		total_s += end_t - start_t;

	c++;

  }

  double averMsecs = total_s / nIter * 1000;
  printf("iterated %d times, average time is %lf ms.\n", nIter, averMsecs);
}

main (int argc, char *argv[]) {


  size_t A_sz, B_sz, C_sz;
  int matArow, matAcol;
  int matBrow, matBcol;
  std::vector<float> matA, matBT;


  /* Read command line. Expect 3 inputs: A, B and B^T 
     in column-major layout*/
  if ((argv[1] == NULL) 
      || (argv[2] == NULL)
      || (argv[3] == NULL))
    {
      fprintf(stderr, "Expecting three input filenames\n");
      exit(-1);
    }

  cl_int clStatus;
  cl_platform_id clPlatform;
  clStatus = clGetPlatformIDs(1,&clPlatform,NULL);
  CHECK_ERROR("clGetPlatformIDs")

  cl_context_properties clCps[3] = {CL_CONTEXT_PLATFORM,(cl_context_properties)clPlatform,0};
  cl_context clContext = clCreateContextFromType(clCps,CL_DEVICE_TYPE_GPU,NULL,NULL,&clStatus);
  CHECK_ERROR("clCreateContextFromType")
   
  cl_device_id clDevice;
  clStatus = clGetDeviceIDs(clPlatform,CL_DEVICE_TYPE_GPU,1,&clDevice,NULL);
  CHECK_ERROR("clGetDeviceIDs")

  cl_command_queue clCommandQueue = clCreateCommandQueue(clContext,clDevice,CL_QUEUE_PROFILING_ENABLE,&clStatus);
  CHECK_ERROR("clCreateCommandQueue")


  const char* clSource[] = {readFile("src/opencl_base/kernel.cl")};
  cl_program clProgram = clCreateProgramWithSource(clContext,1,clSource,NULL,&clStatus);
  CHECK_ERROR("clCreateProgramWithSource")

  char clOptions[50];
  sprintf(clOptions,"");

  clStatus = clBuildProgram(clProgram,1,&clDevice,clOptions,NULL,NULL);
  CHECK_ERROR("clBuildProgram")

  cl_kernel clKernel = clCreateKernel(clProgram,"mysgemmNT",&clStatus);
  CHECK_ERROR("clCreateKernel")

  /* Read in data */

  // load A
  readColMajorMatrixFile(argv[1],
      matArow, matAcol, matA);
  // copy A to device memory
  A_sz = matArow*matAcol*sizeof(float);

  // load B^T
  readColMajorMatrixFile(argv[3],
      matBcol, matBrow, matBT);

  B_sz = matBrow*matBcol*sizeof(float);

  // allocate space for C
  C_sz = matArow*matBcol*sizeof(float);

  // OpenCL memory allocation
  std::vector<float> matC(matArow*matBcol);
  cl_mem dA = clCreateBuffer(clContext,CL_MEM_READ_ONLY,A_sz,NULL,&clStatus);
  CHECK_ERROR("clCreateBuffer")
  cl_mem dB = clCreateBuffer(clContext,CL_MEM_READ_ONLY,B_sz,NULL,&clStatus);
  CHECK_ERROR("clCreateBuffer")
  cl_mem dC = clCreateBuffer(clContext,CL_MEM_WRITE_ONLY,C_sz,NULL,&clStatus);
  CHECK_ERROR("clCreateBuffer")

  // Copy A and B^T into device memory
  clStatus = clEnqueueWriteBuffer(clCommandQueue,dA,CL_FALSE,0,A_sz,&matA.front(),0,NULL,NULL);
  CHECK_ERROR("clEnqueueWriteBuffer")
  clStatus = clEnqueueWriteBuffer(clCommandQueue,dB,CL_FALSE,0,B_sz,&matBT.front(),0,NULL,NULL);
  CHECK_ERROR("clEnqueueWriteBuffer")

  for(int i=0;i<matC.size();i++)
	matC[i] = 0.0f;

  clStatus = clEnqueueWriteBuffer(clCommandQueue,dC,CL_TRUE,0,C_sz,&matC.front(),0,NULL,NULL);
  CHECK_ERROR("clEnqueueWriteBuffer")


  // Use standard sgemm interface
  basicSgemm('N', 'T', matArow, matBcol, matAcol, 1.0f, \
      dA, matArow, dB, matBcol, 0.0f, dC, matArow, clKernel, clCommandQueue);

  if (argv[4]) {
    clEnqueueReadBuffer(clCommandQueue,dC,CL_TRUE,0,C_sz,&matC.front(),0,NULL,NULL);
   
    /* Write C to file */
    writeColMajorMatrixFile(argv[4],
	matArow, matBcol, matC); 
  }



  free((void*)clSource[0]);

  clStatus = clReleaseKernel(clKernel);
  clStatus = clReleaseProgram(clProgram);
  clStatus = clReleaseMemObject(dA);
  clStatus = clReleaseMemObject(dB);
  clStatus = clReleaseMemObject(dC);
  clStatus = clReleaseCommandQueue(clCommandQueue);
  clStatus = clReleaseContext(clContext); 
  
  return 0;
}
