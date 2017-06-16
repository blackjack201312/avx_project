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
#include <omp.h>
#include <sys/time.h>
#include <malloc.h>

#include <parboil.h>

#include "file.h"
#include "computeQ.cc"

int
main (int argc, char *argv[]) {
	int numX, numK;		/* Number of X and K values */
	int original_numK;		/* Number of K values in input file */
	float *kx, *ky, *kz;		/* K trajectory (3D vectors) */
	float *x, *y, *z;		/* X coordinates (3D vectors) */
	float *phiR, *phiI;		/* Phi values (complex) */
	float *phiMag;		/* Magnitude of Phi */
	float *Qr, *Qi;		/* Q signal (complex) */
	struct kValues* kVals;

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

	kVals = (struct kValues*)calloc(numK, sizeof (struct kValues));

	const char* env_iter = getenv("ITER");
	int iteration = (env_iter != NULL) ? atoi(env_iter) : 1;
	printf("[ITERATION NUM]:%d\n", iteration);
	double time = omp_get_wtime();
	#pragma offload target(mic:0)  \
	in(iteration) \
	in(kx,ky,kz:length(numK))  \
	in(x,y,z:length(numX))   \
	in(phiMag:length(numK))  \
	inout(Qr,Qi:length(numX))  \
	in(phiR,phiI:length(numK)) 
	{
		int iter;
		for (iter = 0; iter < iteration; iter++)
		{
			ComputePhiMagCPU(numK, phiR, phiI, phiMag);
			int k;
			ComputeQCPU(numK, numX, kx, ky, kz, phiMag, x, y, z, Qr, Qi);
		}
	}
	 printf("[RUN TIME]:%f\n", omp_get_wtime() - time);

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
	free (kVals);
	free (Qr);
	free (Qi);

	pb_SwitchToTimer(&timers, pb_TimerID_NONE);
	pb_PrintTimerSet(&timers);
	pb_FreeParameters(params);

	return 0;
}
