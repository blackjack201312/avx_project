/*****************************************************************************/
/*IMPORTANT:  READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.         */
/*By downloading, copying, installing or using the software you agree        */
/*to this license.  If you do not agree to this license, do not download,    */
/*install, copy or use the software.                                         */
/*                                                                           */
/*                                                                           */
/*Copyright (c) 2005 Northwestern University                                 */
/*All rights reserved.                                                       */

/*Redistribution of the software in source and binary forms,                 */
/*with or without modification, is permitted provided that the               */
/*following conditions are met:                                              */
/*                                                                           */
/*1       Redistributions of source code must retain the above copyright     */
/*        notice, this list of conditions and the following disclaimer.      */
/*                                                                           */
/*2       Redistributions in binary form must reproduce the above copyright   */
/*        notice, this list of conditions and the following disclaimer in the */
/*        documentation and/or other materials provided with the distribution.*/
/*                                                                            */
/*3       Neither the name of Northwestern University nor the names of its    */
/*        contributors may be used to endorse or promote products derived     */
/*        from this software without specific prior written permission.       */
/*                                                                            */
/*THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS    */
/*IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED      */
/*TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY, NON-INFRINGEMENT AND         */
/*FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL          */
/*NORTHWESTERN UNIVERSITY OR ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT,       */
/*INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES          */
/*(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR          */
/*SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)          */
/*HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,         */
/*STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN    */
/*ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE             */
/*POSSIBILITY OF SUCH DAMAGE.                                                 */
/******************************************************************************/
/*************************************************************************/
/**   File:         kmeans_clustering.c                                 **/
/**   Description:  Implementation of regular k-means clustering        **/
/**                 algorithm                                           **/
/**   Author:  Wei-keng Liao                                            **/
/**            ECE Department, Northwestern University                  **/
/**            email: wkliao@ece.northwestern.edu                       **/
/**                                                                     **/
/**   Edited by: Jay Pisharath                                          **/
/**              Northwestern University.                               **/
/**                                                                     **/
/**   ================================================================  **/
/**																		**/
/**   Edited by: Sang-Ha  Lee											**/
/**				 University of Virginia									**/
/**																		**/
/**   Description:	No longer supports fuzzy c-means clustering;	 	**/
/**					only regular k-means clustering.					**/
/**					Simplified for main functionality: regular k-means	**/
/**					clustering.											**/
/**                                                                     **/
/*************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include "kmeans.h"
#include <omp.h>
#include <malloc.h>
#include <immintrin.h>

#define RANDOM_MAX 2147483647

#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif

extern double wtime(void);
extern int num_omp_threads;

//inline
int find_nearest_point(float  *pt,          /* [nfeatures] */
	int     nfeatures,
	float **pts,         /* [npts][nfeatures] */
	int     npts)
	{
		int index, i;
		float min_dist=FLT_MAX;

		/* find the cluster center id with min distance to pt */
		for (i=0; i<npts; i++) {
			float dist;
			dist = euclid_dist_2(pt, pts[i], nfeatures);  /* no need square root */
			if (dist < min_dist) {
				min_dist = dist;
				index    = i;
			}
		}
		return(index);
	}

	/*----< euclid_dist_2() >----------------------------------------------------*/
	/* multi-dimensional spatial Euclid distance square */
	//inline
	float euclid_dist_2(float *pt1,
		float *pt2,
		int    numdims)
		{
			int i;
			float ans=0.0;

			float sans;
			float *test1, *test2, *test3, *test4;

			__m256 ymm0, ymm1, ymm2;
			//#pragma omp simd reduction (+:ans)
			for (i=0; i<numdims; i+=8)
			{
				ymm1 = _mm256_load_ps(&pt1[i]);
				ymm2 = _mm256_load_ps(&pt2[i]);
				ymm1 = _mm256_sub_ps(ymm1, ymm2);
				ymm0 = _mm256_mul_ps(ymm1, ymm1);
				test1 = (float*)&ymm0;
				test2 = (float*)&ymm1;
				 for(int a = 0; a < 8; a++)
				 printf("itr:%d\ntest1: %f, test2: %f\n", i+a, test1[a], ((pt1[a+i]-pt2[a+i])*(pt1[a+i]-pt2[a+i])));
				// test = (float*)&ymm0;
				// printf("sans: %f, ymm0[0]: %f\n", sans, test[0]);
    		/*reduction*/
				ymm1 = _mm256_hadd_ps(ymm0, ymm0);
				ymm2 = _mm256_hadd_ps(ymm1, ymm1);
				test3 = (float*)&ymm2;
				test4 = (float*)&ymm1;
				for(int a = 0; a < 8; a++)
				printf("itr:%d\ntest3: %f, test4: %f\n", i+a, ((pt1[a+i]-pt2[a+i])*(pt1[a+i]-pt2[a+i])), test3[a]);
				ymm1 = _mm256_permute2f128_ps(ymm2, ymm2, 0x01);
				test1 = (float*)&ymm2;
				test2 = (float*)&ymm1;
				for(int a = 0; a < 8; a++)
				// printf("itr:%d\ntest1: %f, test2: %f\n", a+i, test1[a], test2[a]);
				ymm0 = _mm256_add_ps(ymm1, ymm2);
				test2 = (float*)&ymm0;
				for(int n = 0; n < 8; n++)
				ans += (pt1[n+i]-pt2[n+i]) * (pt1[n+i]-pt2[n+i]);


			}
			printf("Number in vec:%f, ans:%f\n", test2[0], ans);
			//test = (float*)&ymm0;
			//if(test[0] != ans)
			//printf("test[0]: %f, ans: %f\n", test[0], ans);

			return(ans);
		}

		double gettime() {
			struct timeval t;
			gettimeofday(&t,NULL);
			return t.tv_sec+t.tv_usec*1e-6;
		}

		/*----< kmeans_clustering() >---------------------------------------------*/
		float** kmeans_clustering(float **feature,    /* in: [npoints][nfeatures] */
			int     nfeatures,
			int     npoints,
			int     nclusters,
			float   threshold,
			int    *membership,
			int     numThreads) /* out: [npoints] */
			{

				int      i, j, k, n=0, index, loop=0;
				int     *new_centers_len;			/* [nclusters]: no. of points in each cluster */
				float  **new_centers;				/* [nclusters][nfeatures] */
				float  **clusters;					/* out: [nclusters][nfeatures] */
				float    delta;

				double   timing;

				int      nthreads;
				int    **partial_new_centers_len;
				float ***partial_new_centers;

				//const char* env_omp = getenv("OMP_NUM_THREADS");
				//int env_num_omp_threads = (env_omp != NULL) ? atoi(env_omp) : 1;
				nthreads = numThreads;
				omp_set_num_threads(nthreads);
				//printf("[Number of threads]:%d\n", nthreads);

				/* allocate space for returning variable clusters[] */
				clusters    = (float**) memalign(16, nclusters *             sizeof(float*));
				clusters[0] = (float*)  memalign(16, nclusters * nfeatures * sizeof(float));
				for (i=1; i<nclusters; i++)
				clusters[i] = clusters[i-1] + nfeatures;

				/* randomly pick cluster centers */
				//#pragma omp parallel for collapse(2)
				for (i=0; i<nclusters; i++) {
					//n = (int)rand() % npoints;
					for (j=0; j<nfeatures; j++)
					clusters[i][j] = feature[n][j];
					n++;
				}
				//printf("%d\n", nfeatures * npoints - n);
				for (i=0; i<npoints; i++)
				membership[i] = -1;

				/* need to initialize new_centers_len and new_centers[0] to all 0 */
				new_centers_len = (int*) calloc(nclusters, sizeof(int));

				new_centers    = (float**) malloc(nclusters *      sizeof(float*));
				new_centers[0] = (float*)  memalign(16, nclusters * nfeatures * sizeof(float));
				memset(new_centers[0], 0, nclusters * nfeatures * sizeof(float));

				for (i=1; i<nclusters; i++)
				new_centers[i] = new_centers[i-1] + nfeatures;


				partial_new_centers_len    = (int**) memalign(16, nthreads * sizeof(int*));
				partial_new_centers_len[0] = (int*)  calloc(nthreads*nclusters, sizeof(int));
				for (i=1; i<nthreads; i++)
				partial_new_centers_len[i] = partial_new_centers_len[i-1]+nclusters;

				partial_new_centers    =(float***)malloc(nthreads * sizeof(float**));
				partial_new_centers[0] =(float**) malloc(nthreads * nclusters * sizeof(float*));
				int cal = 0;
				for (i=1; i<nthreads; i++){
					partial_new_centers[i] = partial_new_centers[i-1] + nclusters;
					for (j=1; j<nclusters; j++)
					partial_new_centers[i][j] = partial_new_centers[i][j-1] + nfeatures;
				}
				//partial_new_centers[0][0] = (float*)memalign(16, nthreads * nclusters * nfeatures * sizeof(float));
				//memset(partial_new_centers[0][0], 0, nthreads * nclusters * nfeatures * sizeof(float));
				for (i=0; i<nthreads; i++)
				{
					for (j=0; j<nclusters; j++)
					partial_new_centers[i][j] = (float*)calloc(nfeatures, sizeof(float));
				}

				const char* env_iter = getenv("ITER");
				int iteration = (env_iter != NULL) ? atoi(env_iter) : 1;
				//printf("[ITERATION NUM]:%d\n", iteration);

				double start = gettime();
				do {
					delta = 0.0;
//					#pragma omp parallel \
					shared(feature,clusters,membership,partial_new_centers,partial_new_centers_len)
					{
						int tid = omp_get_thread_num();
//						#pragma omp for \
						private(i,j,index) \
						firstprivate(npoints,nclusters,nfeatures) \
						schedule(static) \
						reduction(+:delta)
						for (i=0; i<npoints; i++) {
							/* find the index of nestest cluster centers */
							index = find_nearest_point(feature[i],
								nfeatures,
								clusters,
								nclusters);
								return clusters;
								/* if membership changes, increase delta by 1 */
								if (membership[i] != index) delta += 1.0;

								/* assign the membership to object i */
								membership[i] = index;

								/* update new cluster centers : sum of all objects located
								within */
								partial_new_centers_len[tid][index]++;
								for (j=0; j<nfeatures; j++)
								partial_new_centers[tid][index][j] += feature[i][j];
							}
						} /* end of #pragma omp parallel */

						/* let the main thread perform the array reduction */
						//#pragma omp parallel for
						for (i=0; i<nclusters; i++) {
							#pragma omp simd
							for (j=0; j<nthreads; j++) {
								new_centers_len[i] += partial_new_centers_len[j][i];
								partial_new_centers_len[j][i] = 0.0;
								//#pragma omp simd
								for (k=0; k<nfeatures; k++) {
									new_centers[i][k] += partial_new_centers[j][i][k];
									partial_new_centers[j][i][k] = 0.0;
								}
							}
						}

						/* replace old cluster centers with new_centers */
						for (i=0; i<nclusters; i++) {
							for (j=0; j<nfeatures; j++) {
								if (new_centers_len[i] > 0)
								clusters[i][j] = new_centers[i][j] / new_centers_len[i];
								new_centers[i][j] = 0.0;   /* set back to 0 */
							}
							new_centers_len[i] = 0;   /* set back to 0 */
						}

					} while (loop++ < iteration);

					double end = gettime();

					//printf("Time for %d loops clustering is %lf seconds.\n", iteration, end - start);


					free(new_centers[0]);
					free(new_centers);
					free(new_centers_len);

					return clusters;
				}
