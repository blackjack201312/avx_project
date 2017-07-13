/*
* =====================================================================================
*
*       Filename:  suite.c
*
*    Description:  The main wrapper for the suite
*
*        Version:  1.0
*        Created:  10/22/2009 08:40:34 PM
*       Revision:  none
*       Compiler:  gcc
*
*         Author:  Liang Wang (lw2aw), lw2aw@virginia.edu
*        Company:  CS@UVa
*
* =====================================================================================
*/

#include <stdio.h>
#include <unistd.h>
#include <getopt.h>
#include <stdlib.h>
#include <assert.h>

#include "common.h"

static int do_verify = 0;
int omp_num_threads = 4;

static struct option long_options[] = {
  /* name, has_arg, flag, val */
  {"input", 1, NULL, 'i'},
  {"size", 1, NULL, 's'},
  {"verify", 0, NULL, 'v'},
  {0,0,0,0}
};

double gettime(){
  struct timeval t;
  gettimeofday(&t, NULL);
  return t.tv_sec + t.tv_usec * 1e-6;
}

extern void
lud_omp(float *m, int matrix_dim);

int
main ( int argc, char *argv[] )
{
  int matrix_dim = 32; /* default size */
  int opt, option_index=0;
  func_ret_t ret;
  const char *input_file = NULL;
  float *m, *mm;
  stopwatch sw;


  while ((opt = getopt_long(argc, argv, "::vs:n:i:",
  long_options, &option_index)) != -1 ) {
    switch(opt){
      case 'i':
      input_file = optarg;
      break;
      case 'v':
      do_verify = 1;
      break;
      case 'n':
      omp_num_threads = atoi(optarg);
      break;
      case 's':
      matrix_dim = atoi(optarg); printf("Generate input matrix internally, size =%d\n", matrix_dim); // fprintf(stderr, "Currently not supported, use -i instead\n"); // fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n", argv[0]);
      // exit(EXIT_FAILURE);
      break;
      case '?':
      fprintf(stderr, "invalid option\n");
      break;
      case ':':
      fprintf(stderr, "missing argument\n");
      break;
      default:
      fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n",
      argv[0]);
      exit(EXIT_FAILURE);
    }
  }

  if ( (optind < argc) || (optind == 1)) {
    fprintf(stderr, "Usage: %s [-v] [-n no. of threads] [-s matrix_size|-i input_file]\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  if (input_file) {
    printf("Reading matrix from file %s\n", input_file);
    ret = create_matrix_from_file(&m, input_file, &matrix_dim);
    if (ret != RET_SUCCESS) {
      m = NULL;
      fprintf(stderr, "error create matrix from file %s\n", input_file);
      exit(EXIT_FAILURE);
    }
  }
  else if (matrix_dim) {
    printf("Creating matrix internally size=%d\n", matrix_dim);
    ret = create_matrix(&m, matrix_dim);
    if (ret != RET_SUCCESS) {
      m = NULL;
      fprintf(stderr, "error create matrix internally size=%d\n", matrix_dim);
      exit(EXIT_FAILURE);
    }
  }

  else {
    printf("No input file specified!\n");
    exit(EXIT_FAILURE);
  }

  if (do_verify){
    printf("Before LUD\n");
    /* print_matrix(m, matrix_dim); */
    matrix_duplicate(m, &mm, matrix_dim);
  }

  int i;
  //const char* env_iter = getenv("ITER");
  //int iteration = (env_iter != NULL) ? atoi(env_iter) : 1;
  const char* env_itrs = getenv("ITERS");
  int iteration = (env_itrs != NULL) ? atoi(env_itrs) : 1;
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

  printf("[ITERATION NUM]:%d\n", iteration);



  for (i = -3 ;i<iteration;i++){
    start_t = gettime();
    if(i == 0)  stopwatch_start(&sw);
    lud_omp(m, matrix_dim);
    end_t = gettime();

    if(i == -3)
    {
      c0++;
      total_s0 += end_t - start_t;
    }
    else if(i == -2)
    {
      c1++;
      total_s1 += end_t - start_t;
    }
    else if(i == -1)
    {
      c2++;
      total_s2 += end_t - start_t;
      double tPerIter = total_s2 / c2;
      printf("Sampling from the first itr.\nEstimated time: %lf s.\nEstimated itr: %d.\n", total_s0 / (double)c0, (int)((double)secs / (total_s0 / (double)c0)));
      printf("Sampling from the middle itr.\nEstimated time: %lf s.\nEstimated itr: %d.\n", total_s1 / (double)c1, (int)((double)secs / (total_s1 / (double)c1)));
      printf("Sampling from the last itr.\nEstimated time: %lf s.\n", tPerIter);
      iteration = (int)((double)secs / tPerIter) + 1;
      printf("Adjust %d iterations to meet %d seconds.\n", iteration, secs);
    }
  }
  stopwatch_stop(&sw);
  double averMsecs = get_interval_by_sec(&sw) / iteration * 1000;
  printf("iterated %d times, average time is %lf ms.\n", iteration, averMsecs);
  printf("Time consumed(ms): %lf\n", 1000*get_interval_by_sec(&sw));

  if (do_verify){
    printf("After LUD\n");
    /* print_matrix(m, matrix_dim); */
    printf(">>>Verify<<<<\n");
    lud_verify(mm, m, matrix_dim);
    free(mm);
  }

  free(m);

  return EXIT_SUCCESS;
}				/* ----------  end of function main  ---------- */
