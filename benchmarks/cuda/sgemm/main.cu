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
#include "sgemm_kernel.cu"

// I/O routines
extern bool readColMajorMatrixFile(const char *fn, int &nr_row, int &nr_col, std::vector<float>&v);
extern bool writeColMajorMatrixFile(const char *fn, int, int, std::vector<float>&);

extern "C"
void computeGold(float *, const float*, const float*, unsigned int, unsigned int, unsigned int);

int
main (int argc, char *argv[]) {


  float *dA, *dB, *dC;
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

  // CUDA memory allocation
  std::vector<float> matC(matArow*matBcol);
  cudaMalloc((void**)&dA, A_sz);
  cudaMalloc((void**)&dB, B_sz);
  cudaMalloc((void**)&dC, C_sz);

  // Copy A and B^T into device memory
  cudaMemcpy(dA, &matA.front(), A_sz, cudaMemcpyHostToDevice); 
  cudaMemcpy(dB, &matBT.front(), B_sz, cudaMemcpyHostToDevice); 

for(int i=0;i<num_iter_control;i++){
  // Use standard sgemm interface
  regtileSgemm('N', 'T', matArow, matBcol, matAcol, 1.0f, \
      dA, matArow, dB, matBcol, 0.0f, dC, matArow);
}
  if (argv[4]) {
    cudaMemcpy(&matC.front(), dC, C_sz, cudaMemcpyDeviceToHost);
    /* Write C to file */
    writeColMajorMatrixFile(argv[4],
	matArow, matBcol, matC); 
  }


  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  return 0;
}
