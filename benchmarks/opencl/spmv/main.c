/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

#include "file.h"
#include "gpu_info.h"
#include "ocl.h"
#include "convert_dataset.h"

double gettime()
{
	struct timeval t;
	gettimeofday(&t, NULL);
	return t.tv_sec + t.tv_usec * 1e-6;
}

static int generate_vector(float *x_vector, int dim) 
{	
	srand(54321);
	int i;
	for(i=0;i<dim;i++)
	{
		x_vector[i] = (rand() / (float) RAND_MAX);
	}
	return 0;
}

int main(int argc, char** argv) {
	
	printf("CUDA accelerated sparse matrix vector multiplication****\n");
	printf("Original version by Li-Wen Chang <lchang20@illinois.edu> and Shengzhao Wu<wu14@illinois.edu>\n");
	printf("This version maintained by Chris Rodrigues  ***********\n");
	if ((argv[1] == NULL) || (argv[2] == NULL))
    	{
      		fprintf(stderr, "Expecting one input matrix filename and one input vector filename\n");
      		exit(-1);
    	}

	
	//parameters declaration
	cl_int clStatus;
	cl_platform_id clPlatform;
	clStatus = clGetPlatformIDs(1,&clPlatform,NULL);
	CHECK_ERROR("clGetPlatformIDs")

	cl_context_properties clCps[3] = {CL_CONTEXT_PLATFORM,(cl_context_properties)clPlatform,0};
	
	cl_device_id clDevice;
	clStatus = clGetDeviceIDs(clPlatform,CL_DEVICE_TYPE_GPU,1,&clDevice,NULL);
	CHECK_ERROR("clGetDeviceIDs")

	cl_context clContext = clCreateContextFromType(clCps,CL_DEVICE_TYPE_GPU,NULL,NULL,&clStatus);
	CHECK_ERROR("clCreateContextFromType")

	cl_command_queue clCommandQueue = clCreateCommandQueue(clContext,clDevice,CL_QUEUE_PROFILING_ENABLE,&clStatus);
	CHECK_ERROR("clCreateCommandQueue")

	
	const char* clSource[] = {readFile("src/opencl_base/kernel.cl")};
	cl_program clProgram = clCreateProgramWithSource(clContext,1,clSource,NULL,&clStatus);
	CHECK_ERROR("clCreateProgramWithSource")

	char clOptions[50];
	sprintf(clOptions,"");
	clStatus = clBuildProgram(clProgram,1,&clDevice,clOptions,NULL,NULL);
	CHECK_ERROR("clBuildProgram")

	cl_kernel clKernel = clCreateKernel(clProgram,"spmv_jds_naive",&clStatus);
	CHECK_ERROR("clCreateKernel")

	int len;
	int depth;
	int dim;
	int pad=32;
	int nzcnt_len;
	
	//host memory allocation
	//matrix
	float *h_data;
	int *h_indices;
	int *h_ptr;
	int *h_perm;
	int *h_nzcnt;
	//vector
	float *h_Ax_vector;
	float *h_x_vector;
	
	//device memory allocation
	//matrix
	cl_mem d_data;
	cl_mem d_indices;
	cl_mem d_ptr;
	cl_mem d_perm;
	cl_mem d_nzcnt;

	//vector
	cl_mem d_Ax_vector;
	cl_mem d_x_vector;
	
	cl_mem jds_ptr_int;
	cl_mem sh_zcnt_int;

    	//load matrix from files
	//inputData(parameters->inpFiles[0], &len, &depth, &dim,&nzcnt_len,&pad,
	//    &h_data, &h_indices, &h_ptr,
	//    &h_perm, &h_nzcnt);
	int col_count;
	coo_to_jds(
		argv[1], // bcsstk32.mtx, fidapm05.mtx, jgl009.mtx
		1, // row padding
		pad, // warp size
		1, // pack size
		1, // is mirrored?
		0, // binary matrix
		1, // debug level [0:2]
		&h_data, &h_ptr, &h_nzcnt, &h_indices, &h_perm,
		&col_count, &dim, &len, &nzcnt_len, &depth
	);
	
	h_Ax_vector=(float*)malloc(sizeof(float)*dim);	
	h_x_vector=(float*)malloc(sizeof(float)*dim);	
	
  input_vec( argv[2],h_x_vector,dim);

    	OpenCLDeviceProp clDeviceProp;
//	clStatus = clGetDeviceInfo(clDevice,CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV,sizeof(cl_uint),&(clDeviceProp.major),NULL);
	//CHECK_ERROR("clGetDeviceInfo")
//	clStatus = clGetDeviceInfo(clDevice,CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV,sizeof(cl_uint),&(clDeviceProp.minor),NULL);
  //      CHECK_ERROR("clGetDeviceInfo")
	clStatus = clGetDeviceInfo(clDevice,CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(cl_uint),&(clDeviceProp.multiProcessorCount),NULL);
        CHECK_ERROR("clGetDeviceInfo")
	
	//memory allocation
	d_data = clCreateBuffer(clContext,CL_MEM_READ_ONLY,len*sizeof(float),NULL,&clStatus);
	CHECK_ERROR("clCreateBuffer")
	d_indices = clCreateBuffer(clContext,CL_MEM_READ_ONLY,len*sizeof(int),NULL,&clStatus);
	CHECK_ERROR("clCreateBuffer")
	d_perm = clCreateBuffer(clContext,CL_MEM_READ_ONLY,dim*sizeof(int),NULL,&clStatus);
	CHECK_ERROR("clCreateBuffer")
	d_x_vector = clCreateBuffer(clContext,CL_MEM_READ_ONLY,dim*sizeof(float),NULL,&clStatus);
	CHECK_ERROR("clCreateBuffer")
	d_Ax_vector = clCreateBuffer(clContext,CL_MEM_WRITE_ONLY,dim*sizeof(float),NULL,&clStatus);
	CHECK_ERROR("clCreateBuffer")

	jds_ptr_int = clCreateBuffer(clContext,CL_MEM_READ_ONLY,5000*sizeof(int),NULL,&clStatus);
	CHECK_ERROR("clCreateBuffer")
	sh_zcnt_int = clCreateBuffer(clContext,CL_MEM_READ_ONLY,5000*sizeof(int),NULL,&clStatus);
	CHECK_ERROR("clCreateBuffer")

	clMemSet(clCommandQueue,d_Ax_vector,0,dim*sizeof(float));
	
	//memory copy
	clStatus = clEnqueueWriteBuffer(clCommandQueue,d_data,CL_FALSE,0,len*sizeof(float),h_data,0,NULL,NULL);
	CHECK_ERROR("clEnqueueWriteBuffer")
	clStatus = clEnqueueWriteBuffer(clCommandQueue,d_indices,CL_FALSE,0,len*sizeof(int),h_indices,0,NULL,NULL);
	CHECK_ERROR("clEnqueueWriteBuffer")
	clStatus = clEnqueueWriteBuffer(clCommandQueue,d_perm,CL_FALSE,0,dim*sizeof(int),h_perm,0,NULL,NULL);
	CHECK_ERROR("clEnqueueWriteBuffer")
	clStatus = clEnqueueWriteBuffer(clCommandQueue,d_x_vector,CL_FALSE,0,dim*sizeof(int),h_x_vector,0,NULL,NULL);
	CHECK_ERROR("clEnqueueWriteBuffer")

	clStatus = clEnqueueWriteBuffer(clCommandQueue,jds_ptr_int,CL_FALSE,0,depth*sizeof(int),h_ptr,0,NULL,NULL);
	CHECK_ERROR("clEnqueueWriteBuffer")
	clStatus = clEnqueueWriteBuffer(clCommandQueue,sh_zcnt_int,CL_TRUE,0,nzcnt_len*sizeof(int),h_nzcnt,0,NULL,NULL);
	CHECK_ERROR("clEnqueueWriteBuffer")
	
	
	size_t grid;
	size_t block;

	compute_active_thread(&block,&grid,nzcnt_len,pad,clDeviceProp.major,clDeviceProp.minor,clDeviceProp.multiProcessorCount);
//  printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!grid is %d and block is %d=\n",grid,block);
//  printf("!!! dim is %d\n",dim);

	clStatus = clSetKernelArg(clKernel,0,sizeof(cl_mem),&d_Ax_vector);
	CHECK_ERROR("clSetKernelArg")
	clStatus = clSetKernelArg(clKernel,1,sizeof(cl_mem),&d_data);
	CHECK_ERROR("clSetKernelArg")
	clStatus = clSetKernelArg(clKernel,2,sizeof(cl_mem),&d_indices);
	CHECK_ERROR("clSetKernelArg")
	clStatus = clSetKernelArg(clKernel,3,sizeof(cl_mem),&d_perm);
	CHECK_ERROR("clSetKernelArg")
	clStatus = clSetKernelArg(clKernel,4,sizeof(cl_mem),&d_x_vector);
	CHECK_ERROR("clSetKernelArg")
	clStatus = clSetKernelArg(clKernel,5,sizeof(int),&dim);
	CHECK_ERROR("clSetKernelArg")

	clStatus = clSetKernelArg(clKernel,6,sizeof(cl_mem),&jds_ptr_int);
	CHECK_ERROR("clSetKernelArg")
	clStatus = clSetKernelArg(clKernel,7,sizeof(cl_mem),&sh_zcnt_int);
        CHECK_ERROR("clSetKernelArg")

	//main execution

	const char* env_itrs = getenv("ITERS");
	int nIter = (env_itrs != NULL) ? atoi(env_itrs) : 1;
	const char* env_secs = getenv("SECS");
	int secs = (env_secs != NULL) ? atoi(env_secs) : 10;
	int timeRestrict = (env_secs != NULL) ? 1 : 0;

	double start_t;
	double end_t;
	double total_s = 0;
	int c = 0;


	int i;
	for (i=-1000; i<nIter; i++)
	{
		start_t = gettime();

		clStatus = clEnqueueNDRangeKernel(clCommandQueue,clKernel,1,NULL,&grid,&block,0,NULL,NULL);
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
				nIter = ceil((double)secs / tPerIter) + 1;
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
	
	//HtoD memory copy
	clStatus = clEnqueueReadBuffer(clCommandQueue,d_Ax_vector,CL_TRUE,0,dim*sizeof(float),h_Ax_vector,0,NULL,NULL);
	CHECK_ERROR("clEnqueueReadBuffer")	

	clStatus = clReleaseKernel(clKernel);
	clStatus = clReleaseProgram(clProgram);

	clStatus = clReleaseMemObject(d_data);
	clStatus = clReleaseMemObject(d_indices);
        clStatus = clReleaseMemObject(d_perm);
	clStatus = clReleaseMemObject(d_nzcnt);
        clStatus = clReleaseMemObject(d_x_vector);
	clStatus = clReleaseMemObject(d_Ax_vector);
	CHECK_ERROR("clReleaseMemObject")

	clStatus = clReleaseCommandQueue(clCommandQueue);
	clStatus = clReleaseContext(clContext);	
	
	if (argv[3]) {
		outputData(argv[3],h_Ax_vector,dim);
	}

	
	free((void*)clSource[0]);
	
	free (h_data);
	free (h_indices);
	free (h_ptr);
	free (h_perm);
	free (h_nzcnt);
	free (h_Ax_vector);
	free (h_x_vector);

	return 0;
}
