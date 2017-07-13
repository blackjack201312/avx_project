/***************************************************************************
*cr
*cr            (C) Copyright 2007 The Board of Trustees of the
*cr                        University of Illinois
*cr                         All Rights Reserved
*cr
***************************************************************************/

/*
* C code for creating the Q data structure for fast convolution-based
* Hessian multiplication for arbitrary k-space trajectories.
*
* Inputs:
* kx - VECTOR of kx values, same length as ky and kz
* ky - VECTOR of ky values, same length as kx and kz
* kz - VECTOR of kz values, same length as kx and ky
* x  - VECTOR of x values, same length as y and z
* y  - VECTOR of y values, same length as x and z
* z  - VECTOR of z values, same length as x and y
* phi - VECTOR of the Fourier transform of the spatial basis
*      function, evaluated at [kx, ky, kz].  Same length as kx, ky, and kz.
*
* recommended g++ options:
*  -O3 -lm -ffast-math -funroll-all-loops
*/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <malloc.h>
#include <omp.h>

#include <parboil.h>

#include "file.h"
#include "computeQ.cc"

inline double gettime(){
  struct timeval t;
  gettimeofday(&t, NULL);
  return t.tv_sec + t.tv_usec * 1e-6;
}

int
main (int argc, char *argv[]) {
  int numX, numK;		/* Number of X and K values */
  int original_numK;		/* Number of K values in input file */
  float *kx, *ky, *kz;		/* K trajectory (3D vectors) */
  float *x, *y, *z;		/* X coordinates (3D vectors) */
  float *phiR, *phiI;		/* Phi values (complex) */
  float *phiMag;		/* Magnitude of Phi */
  float *Qr, *Qi;		/* Q signal (complex) */
  //struct kValues* kVals;

  struct pb_Parameters *params;
  struct pb_TimerSet timers;

  pb_InitializeTimerSet(&timers);

  /* Read command line */
  params = pb_ReadParameters(&argc, argv);
  if ((params->inpFiles[0] == NULL) || (params->inpFiles[1] != NULL))
  {
    fprintf(stderr, "Expecting one input filename\n");
    exit(-1);
  }

  /* Read in data */
  pb_SwitchToTimer(&timers, pb_TimerID_IO);
  inputData(params->inpFiles[0],
    &original_numK, &numX,
    &kx, &ky, &kz,
    &x, &y, &z,
    &phiR, &phiI);

    /* Reduce the number of k-space samples if a number is given
    * on the command line */
    if (argc < 2)
    numK = original_numK;
    else
    {
      int inputK;
      char *end;
      inputK = strtol(argv[1], &end, 10);
      if (end == argv[1])
      {
        fprintf(stderr, "Expecting an integer parameter\n");
        exit(-1);
      }

      numK = MIN(inputK, original_numK);
    }

    printf("%d pixels in output; %d samples in trajectory; using %d samples\n",
    numX, original_numK, numK);

    pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

    /* Create CPU data structures */
    createDataStructsCPU(numK, numX, &phiMag, &Qr, &Qi);

    const char* env_itrs = getenv("ITERS");
    int nIter = (env_itrs != NULL) ? atoi(env_itrs) : 1;
    const char* env_secs = getenv("SECS");
    int secs = (env_secs != NULL) ? atoi(env_secs) : -1;

    double start_t, end_t;
    double total_s0, total_s1, total_s2 = 0;
    int c0, c1, c2 = 0;

    if(secs == -1){
      fprintf(stderr, "You must set a time larger than 0 seconds!\n");
      return 1;

    }
    for(int i = -30; i < nIter; i++)
    {
      start_t = gettime();

      ComputePhiMagCPU(numK, phiR, phiI, phiMag);

      //kVals = (struct kValues*)calloc(numK, sizeof (struct kValues));
      //Kx = (float*)memalign(16, numK * sizeof(float));
      //ky = (float*)memalign(16, numK * sizeof(float));
      //Kz = (float*)memalign(16, numK * sizeof(float));
      //phimag = (float*)memalign(16, numK * sizeof(float));
      //
      //  int k;
      //  #pragma omp parallel for simd
      //  for (k = 0; k < numK; k++) {
      //    Kx = kx[k];
      //    Ky = ky[k];
      //    Kz = kz[k];
      //    PhiMag = phiMag[k];
      //  }
      ComputeQCPU(numK, numX, kx, ky, kz, phiMag, x, y, z, Qr, Qi);

      end_t = gettime();

      if(i < -20)
      {
        c0++;
        total_s0 += end_t - start_t;
      }
      else if(i < -10 && i >= -20)
      {
        c1++;
        total_s1 += end_t - start_t;
      }
      else if(i >= -10)
      {
        c2++;
        total_s2 += end_t - start_t;
        if(i == -1)
        {
          double tPerIter = total_s2 / c2;
          printf("Sampling from the first 10 itrs.\nEstimated time: %lf s.\nEstimated itr: %d.\n", total_s0 / (double)c0, (int)((double)secs / (total_s0 / (double)c0)));
          printf("Sampling from the middle 10 itrs.\nEstimated time: %lf s.\nEstimated itr: %d.\n", total_s1 / (double)c1, (int)((double)secs / (total_s1 / (double)c1)));
          printf("Sampling from the last 10 itrs.\nEstimated time: %lf s.\n", tPerIter);
          nIter = (int)((double)secs / tPerIter) + 1;
          printf("Adjust %d iterations to meet %d seconds.\n", nIter, secs);
          c2 = 0;
          total_s2 = 0;
        }
      }


    }
    double averMsecs = total_s2 / nIter * 1000;
    printf("itered %d times, average time is %lf ms.\n", nIter, averMsecs);
    if (params->outFile)
    {
      /* Write Q to file */
      pb_SwitchToTimer(&timers, pb_TimerID_IO);
      outputData(params->outFile, Qr, Qi, numX);
      pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
    }

    free (kx);
    free (ky);
    free (kz);
    free (x);
    free (y);
    free (z);
    free (phiR);
    free (phiI);
    free (phiMag);
    //  free (kVals);
    free (Qr);
    free (Qi);

    pb_SwitchToTimer(&timers, pb_TimerID_NONE);
    pb_PrintTimerSet(&timers);
    pb_FreeParameters(params);

    return 0;
  }
