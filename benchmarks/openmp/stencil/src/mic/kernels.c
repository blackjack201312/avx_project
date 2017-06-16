/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "common.h"

void cpu_stencil(float c0, float c1, float *A0, float * Anext,const int nx, const int ny, const int nz, int iteration)
{
	const char* env_iter = getenv("ITER");
	int iterations = (env_iter != NULL) ? atoi(env_iter) : 1;
	printf("[ITERATION NUM]:%d\n", iterations);
	double time = omp_get_wtime();
	#pragma offload target(mic:0) inout(A0:length(nx*ny*nz)) inout(Anext:length(nx*ny*nz))// into(Anext))
	{
		int i, t;  
		int j,k;
		for (t = 0; t < iterations; t++)
		{
			#pragma omp parallel for collapse(3)
			#pragma ivdep
			for(i=1;i<nx-1;i++)
			{
				for(j=1;j<ny-1;j++)
				{
					for(k=1;k<nz-1;k++)
					{
						//i      #pragma omp critical
						Anext[Index3D (nx, ny, i, j, k)] = 
							(A0[Index3D (nx, ny, i, j, k + 1)] +
							 A0[Index3D (nx, ny, i, j, k - 1)] +
							 A0[Index3D (nx, ny, i, j + 1, k)] +
							 A0[Index3D (nx, ny, i, j - 1, k)] +
							 A0[Index3D (nx, ny, i + 1, j, k)] +
							 A0[Index3D (nx, ny, i - 1, j, k)])*c1
							- A0[Index3D (nx, ny, i, j, k)]*c0;
					}
				}
			}
			float *temp = A0;
			A0 = Anext;
			Anext = temp;
		}
	}
	printf("[RUN TIME]:%f\n", omp_get_wtime() - time);
}


