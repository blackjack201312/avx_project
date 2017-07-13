/***************************************************************************
*cr
*cr            (C) Copyright 2010 The Board of Trustees of the
*cr                        University of Illinois
*cr                         All Rights Reserved
*cr
***************************************************************************/

#include <parboil.h>
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <sys/time.h>

#include "file.h"
#include "convert_dataset.h"

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

/*
void jdsmv(int height, int len, float* value, int* perm, int* jds_ptr, int* col_index, float* vector,
float* result){
int i;
int col,row;
int row_index =0;
int prem_indicator=0;
for (i=0; i<len; i++){
if (i>=jds_ptr[prem_indicator+1]){
prem_indicator++;
row_index=0;
}
if (row_index<height){
col = col_index[i];
row = perm[row_index];
result[row]+=value[i]*vector[col];
}

row_index++;
}
return;
}
*/
double gettime(){
  struct timeval t;
  gettimeofday(&t, NULL);
  return t.tv_sec + t.tv_usec * 1e-6;
}


int main(int argc, char** argv) {
  struct pb_TimerSet timers;
  struct pb_Parameters *parameters;



  printf("CPU-based sparse matrix vector multiplication****\n");
  parameters = pb_ReadParameters(&argc, argv);
  if ((parameters->inpFiles[0] == NULL) || (parameters->inpFiles[1] == NULL))
  {
    fprintf(stderr, "Expecting two input filenames\n");
    exit(-1);
  }


  pb_InitializeTimerSet(&timers);
  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  //parameters declaration
  int len;
  int depth;
  int dim;
  int pad=1;
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


  //load matrix from files
  pb_SwitchToTimer(&timers, pb_TimerID_IO);
  //inputData(parameters->inpFiles[0], &len, &depth, &dim,&nzcnt_len,&pad,
  //    &h_data, &h_indices, &h_ptr,
  //    &h_perm, &h_nzcnt);

  //printf("Ready to transform matrix!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");

  int col_count;
  coo_to_jds(
    parameters->inpFiles[0], // bcsstk32.mtx, fidapm05.mtx, jgl009.mtx
    1, // row padding
    pad, // warp size
    1, // pack size
    1, // is mirrored?
    0, // binary matrix
    1, // debug level [0:2]
    &h_data, &h_ptr, &h_nzcnt, &h_indices, &h_perm,
    &col_count, &dim, &len, &nzcnt_len, &depth
  );

  printf("Transform completed!\n");

  h_Ax_vector=(float*)memalign(16, sizeof(float)*dim);
  printf("Matrix allocation successful!\n");
  h_x_vector=(float*)memalign(16, sizeof(float)*dim);
  if(h_x_vector != NULL)
  printf("Vector allocation successful!\n");
  //generate_vector(h_x_vector, dim);
  input_vec(parameters->inpFiles[1],h_x_vector,dim);

  printf("Checked!\n");

  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  printf("Ready for execution\n");

  const char* env_itrs = getenv("ITERS");
  int nIter = (env_itrs != NULL) ? atoi(env_itrs) : 1;
  const char* env_secs = getenv("SECS");
  int secs = (env_secs != NULL) ? atoi(env_secs) : -1;
  //int timeRestrict = (env_secs != NULL) ? 1 : 0;

  double start_t;
  double end_t;
  double total_s0, total_s1, total_s2 = 0.0;
  //double total_s0, total_s1, total_s2;
  //total_s0 = total_s1 = total_s2 = 0;
  int c0, c1, c2;
  c0 = 0;
  c1 = 0;
  c2 = 0;

  if(secs == -1){
    fprintf(stderr, "You must set a time larger than 0 seconds!\n");
    return 1;
  }

  int p, i;
  //main execution
  for(p=-300;p<nIter;p++)
  {

    start_t = gettime();

    #pragma omp parallel for
    for (i = 0; i < dim; i++) {
      int k;
      float sum = 0.0f;
      //int  bound = h_nzcnt[i / 32];
      int  bound = h_nzcnt[i];
      //#pragma omp simd
      for(k=0;k<bound;k++ ) {
        int j = h_ptr[k] + i;
        int in = h_indices[j];

        float d = h_data[j];
        float t = h_x_vector[in];

        sum += d*t;
      }
      //  #pragma omp critical
      h_Ax_vector[h_perm[i]] = sum;
    }
    end_t = gettime();
    if(p < -200)
    {
      c0++;
      total_s0 += end_t - start_t;
    }
    else if(p < -100 && i >= -200)
    {
      c1++;
      total_s1 += end_t - start_t;
    }
    else if(p >= -100)
    {
      c2++;
      total_s2 += end_t - start_t;
      if (p == -1)
      {
        double tPerIter = total_s2 / c2;
        printf("Sampling from the first 100 itrs.\nEstimated time: %lf s.\nEstimated itr: %d.\n", total_s0 / (double)c0, (int)((double)secs / (total_s0 / (double)c0)));
        printf("Sampling from the middle 100 itrs.\nEstimated time: %lf s.\nEstimated itr: %d.\n", total_s1 / (double)c1, (int)((double)secs / (total_s1 / (double)c1)));
        printf("Sampling from the last 100 itrs.\nEstimated time: %lf s.\n", tPerIter);
        nIter = int((double)secs / tPerIter) + 1;
        printf("Adjust %d iterations to meet %d seconds.\n", nIter, secs);
        c2 = 0;
        total_s2=0;
      }
    }
  }

  double averMsecs = total_s2 / nIter * 1000;
  printf("iterated %d times, average time is %lf ms.\n", nIter, averMsecs);

  if (parameters->outFile) {
    pb_SwitchToTimer(&timers, pb_TimerID_IO);
    outputData(parameters->outFile,h_Ax_vector,dim);

  }
  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  free (h_data);
  free (h_indices);
  free (h_ptr);
  free (h_perm);
  free (h_nzcnt);
  free (h_Ax_vector);
  free (h_x_vector);
  pb_SwitchToTimer(&timers, pb_TimerID_NONE);

  pb_PrintTimerSet(&timers);
  pb_FreeParameters(parameters);

  return 0;

}
