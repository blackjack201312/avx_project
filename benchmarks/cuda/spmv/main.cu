
#include <stdio.h>
#include <stdlib.h>

#include "file.h"
#include "gpu_info.h"
#include "spmv_jds.h"
#include "jds_kernels.cu"
#include "convert_dataset.h"


/*
static int generate_vector(float *x_vector, int dim) 
{	
	srand(54321);	
	for(int i=0;i<dim;i++)
	{
		x_vector[i] = (rand() / (float) RAND_MAX);
	}
	return 0;
}
*/

int main(int argc, char** argv) {
	
	
	
	
	
	printf("CUDA accelerated sparse matrix vector multiplication****\n");
	printf("Original version by Li-Wen Chang <lchang20@illinois.edu> and Shengzhao Wu<wu14@illinois.edu>\n");
	printf("This version maintained by Chris Rodrigues  ***********\n");
	if ((argv[1] == NULL) || (argv[2] == NULL))
    {
      fprintf(stderr, "Expecting two input filenames\n");
      exit(-1);
    }


  ///////////////////////////////
  // Running Control Added By Roy
  ///////////////////////////////
  cudaDeviceProp prop;
  int dev_id = atoi(argv[1]);
  int num_iter_control = atoi(argv[2]);
  printf("Device ID is %d, Loop is %d \n",dev_id,num_iter_control);
  printf("Choosing CUDA Device....\n");
  cudaError_t set_result = cudaSetDevice(dev_id);
  printf("Set Result is: %s\n",cudaGetErrorString(set_result));
  cudaGetDevice(&dev_id);
  cudaGetDeviceProperties(&prop, dev_id);
  printf("Name:                     %s\n", prop.name);
  ///////////////////////////////
  // End of Running Control
  ///////////////////////////////

	


	
	//parameters declaration
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
	float *d_data;
	int *d_indices;
	int *d_ptr;
	int *d_perm;
	int *d_nzcnt;
	//vector
	float *d_Ax_vector;
    float *d_x_vector;
	
    //load matrix from files
	//inputData(parameters->inpFiles[0], &len, &depth, &dim,&nzcnt_len,&pad,
	//    &h_data, &h_indices, &h_ptr,
	//    &h_perm, &h_nzcnt);
	int col_count;
	coo_to_jds(
		argv[1], // bcsstk32.mtx, fidapm05.mtx, jgl009.mtx
		1, // row padding
		pad, // warp size, IMPORTANT: change in kernel as well
		1, // pack size
		1, // is mirrored?
		0, // binary matrix
		1, // debug level [0:2]
		&h_data, &h_ptr, &h_nzcnt, &h_indices, &h_perm,
		&col_count, &dim, &len, &nzcnt_len, &depth
	);
	

  h_Ax_vector=(float*)malloc(sizeof(float)*dim); 
  // memset(h_Ax_vector, 0, sizeof(float)*dim);
  h_x_vector=(float*)malloc(sizeof(float)*dim);
  input_vec( argv[2],h_x_vector,dim);

	
	cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev_id);
	
	
	//memory allocation
	cudaMalloc((void **)&d_data, len*sizeof(float));
	cudaMalloc((void **)&d_indices, len*sizeof(int));
	cudaMalloc((void **)&d_ptr, depth*sizeof(int));
	cudaMalloc((void **)&d_perm, dim*sizeof(int));
	cudaMalloc((void **)&d_nzcnt, nzcnt_len*sizeof(int));
	cudaMalloc((void **)&d_x_vector, dim*sizeof(float));
	cudaMalloc((void **)&d_Ax_vector,dim*sizeof(float));
	cudaMemset( (void *) d_Ax_vector, 0, dim*sizeof(float));
	
	//memory copy
	cudaMemcpy(d_data, h_data, len*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_indices, h_indices, len*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_perm, h_perm, dim*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x_vector, h_x_vector, dim*sizeof(float), cudaMemcpyHostToDevice);
	// cudaMemcpy(d_Ax_vector, h_Ax_vector, dim*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(jds_ptr_int, h_ptr, depth*sizeof(int));
	cudaMemcpyToSymbol(sh_zcnt_int, h_nzcnt,nzcnt_len*sizeof(int));
        CUERR
    cudaThreadSynchronize();
	unsigned int grid;
	unsigned int block;
    compute_active_thread(&block, &grid,nzcnt_len,pad, deviceProp.major,deviceProp.minor,
					deviceProp.warpSize,deviceProp.multiProcessorCount);

    cudaFuncSetCacheConfig(spmv_jds, cudaFuncCachePreferL1);

	//main execution
	for(int jj = 0;jj<num_iter_control;jj++){
	spmv_jds<<<grid, block>>>(d_Ax_vector,
  				d_data,d_indices,d_perm,
				d_x_vector,d_nzcnt,dim);
							
	cudaThreadSynchronize();
    CUERR // check and clear any existing errors
	
	}
	cudaThreadSynchronize();
	
	//HtoD memory copy
	cudaMemcpy(h_Ax_vector, d_Ax_vector,dim*sizeof(float), cudaMemcpyDeviceToHost);	

	cudaThreadSynchronize();

	cudaFree(d_data);
    cudaFree(d_indices);
    cudaFree(d_ptr);
	cudaFree(d_perm);
    cudaFree(d_nzcnt);
    cudaFree(d_x_vector);
	cudaFree(d_Ax_vector);
 
	if (argv[3]) {
		outputData(argv[3],h_Ax_vector,dim);
		
	}
	
	free (h_data);
	free (h_indices);
	free (h_ptr);
	free (h_perm);
	free (h_nzcnt);
	free (h_Ax_vector);
	free (h_x_vector);

	return 0;

}
