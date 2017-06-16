/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/* 
 * Base C implementation of MM
 */

#include <stdio.h>
#include <iostream>
#include <omp.h>
#include <stdlib.h>


void basicSgemm( char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc )
{
	if ((transa != 'N') && (transa != 'n')) {
		std::cerr << "unsupported value of 'transa' in regtileSgemm()" << std::endl;
		return;
	}

	if ((transb != 'T') && (transb != 't')) {
		std::cerr << "unsupported value of 'transb' in regtileSgemm()" << std::endl;
		return;
	}
	
	const char* env_iter = getenv("ITER");
	int iteration = (env_iter != NULL) ? atoi(env_iter) : 1;
	printf("[ITERATION NUM]:%d\n", iteration);
	double time = omp_get_wtime();
	#pragma offload target(mic:0) in(m,n,k,lda,ldb,ldc,iteration) in(A:length(m*k)) in(B:length(k*n)) out(C:length(m*n))
	{
		for(int iter = 0; iter < iteration; iter++)
		{
			#pragma omp parallel for collapse (2) 
			for (int mm = 0; mm < m; ++mm) 
			{
				for (int nn = 0; nn < n; ++nn) 
				{
					float c = 0.0f;
					for (int i = 0; i < k; ++i) 
					{
						float a = A[mm + i * lda]; 
						float b = B[nn + i * ldb];
						c += a * b;
					}
					C[mm+nn*ldc] = C[mm+nn*ldc] * beta + alpha * c;
				}
			}
		}
	}
	printf("[RUN TIME]:%f\n", omp_get_wtime() - time);
}
