#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>

#include "backprop.h"

////////////////////////////////////////////////////////////////////////////////

extern void bpnn_layerforward(float *l1, float *l2, float **conn, int n1, int n2);

extern void bpnn_output_error(float *delta, float *target, float *output, int nj, float *err);

extern void bpnn_hidden_error(float *delta_h, int nh, float *delta_o, int no, float **who, float *hidden, float *err);

extern void bpnn_adjust_weights(float *delta, int ndelta, float *ly, int nly, float **w, float **oldw);


extern int setup(int argc, char** argv);

extern float **alloc_2d_dbl(int m, int n);

extern float squash(float x);

double gettime() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec+t.tv_usec*1e-6;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv)
{
  setup(argc, argv);
}


void bpnn_train_kernel(BPNN *net, float *eo, float *eh)
{
  int in, hid, out;
  float out_err, hid_err;

  in = net->input_n;
  hid = net->hidden_n;
  out = net->output_n;

  printf("Performing CPU computation\n");

  int i;
  // const char* env_iter = getenv("ITER");
  // int iteration = (env_iter != NULL) ? atoi(env_iter) : 1;
  // printf("[ITERATION NUM]:%d\n", iteration);
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

  if(secs == -1) {
    fprintf(stderr, "You must set a time larger than 0 seconds!\n");
    return 1;
  }

  for (int i = -60; i<nIter; i++) {

    start_t = gettime();

    bpnn_layerforward(net->input_units, net->hidden_units,net->input_weights, in, hid);
    bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights, hid, out);
    bpnn_output_error(net->output_delta, net->target, net->output_units, out, &out_err);
    bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out, net->hidden_weights, net->hidden_units, &hid_err);
    bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid, net->hidden_weights, net->hidden_prev_weights);
    bpnn_adjust_weights(net->hidden_delta, hid, net->input_units, in, net->input_weights, net->input_prev_weights);

    end_t = gettime();
    if(i < -40)
    {
      c0++;
      total_s0 += end_t - start_t;
    }
    else if(i < -20 && i >= -40)
    {
      c1++;
      total_s1 += end_t - start_t;
    }
    else if(i >= -40)
    {
      c2++;
      total_s2 += end_t - start_t;

      if (i == -1)
      {
        double tPerIter = total_s2 / c2;
        printf("Sampling from the first 20 itrs.\nEstimated time: %lf s.\nEstimated itr: %d.\n", total_s0 / (double)c0, (int)((double)secs / (total_s0 / (double)c0)));
        printf("Sampling from the middle 20 itrs.\nEstimated time: %lf s.\nEstimated itr: %d.\n", total_s1 / (double)c1, (int)((double)secs / (total_s1 / (double)c1)));
        printf("Sampling from the last 20 itrs.\nEstimated time: %lf s.\n", tPerIter);
        nIter = (int)((double)secs / tPerIter) + 1;
        printf("Adjust %d iterations to meet %d seconds.\n", nIter, secs);
        c2 = 0;
        total_s2=0;
      }
    }
  }
  double averMsecs = total_s2 / nIter * 1000;
  printf("Finish %d iterations and spent %lf secs!\n", nIter, total_s2);
  printf("average time is %lf ms.\n", averMsecs);
}
