/***************************************************************************
* cr
* cr            (C) Copyright 2010 The Board of Trustees of the
* cr                        University of Illinois
* cr                         All Rights Reserved
* cr
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

extern void basicSgemm( char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc );

// I/O routines
extern bool readColMajorMatrixFile(const char *fn, int &nr_row, int &nr_col, std::vector<float>&v);
extern bool writeColMajorMatrixFile(const char *fn, int, int, std::vector<float>&);

double gettime(){
        struct timeval t;
        gettimeofday(&t, NULL);
        return t.tv_sec + t.tv_usec * 1e-6;
}


int
main (int argc, char *argv[]) {


        int matArow, matAcol;
        int matBrow, matBcol;
        std::vector<float> matAT, matB;


        /* Read command line. Expect 3 inputs: A, B and B^T
           in column-major layout*/
        // params = pb_ReadParameters(&argc, argv);
        if ((argv[1] == NULL)
            || (argv[2] == NULL)
            || (argv[3] == NULL))
        {
                fprintf(stderr, "Expecting three input filenames\n");
                exit(-1);
        }


        // load A^T
        readColMajorMatrixFile(argv[1],
                               matAcol, matArow, matAT);

        // load B
        readColMajorMatrixFile(argv[3],
                               matBrow, matBcol, matB);


        // allocate space for C
        std::vector<float> matC(matArow*matBcol);

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

        for (int i = -30; i<nIter; i++) {

                start_t = gettime();

                // Use standard sgemm interface
                basicSgemm('T', 'N', matArow, matBcol, matAcol, 1.0f,
                           &matAT.front(), matArow, &matB.front(), matBcol, 0.0f, &matC.front(),
                           matArow);

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

                        if (i == -1)
                        {
                                double tPerIter = total_s2 / c2;
                                printf("Sampling from the first 10 itrs.\nEstimated time: %lf s.\nEstimated itr: %d.\n", total_s0 / (double)c0, (int)((double)secs / (total_s0 / (double)c0)));
                                printf("Sampling from the middle 10 itrs.\nEstimated time: %lf s.\nEstimated itr: %d.\n", total_s1 / (double)c1, (int)((double)secs / (total_s1 / (double)c1)));
                                printf("Sampling from the last 10 itrs.\nEstimated time: %lf s.\n", tPerIter);
                                nIter = (int)((double)secs / tPerIter) + 1;
                                printf("Adjust %d iterations to meet %d seconds.\n", nIter, secs);
                                c2 = 0;
                                total_s2=0;
                        }
                }
        }

        double averMsecs = total_s2 / nIter * 1000;
        printf("iterated %d times, average time is %lf ms.\n", nIter, averMsecs);



        if (argv[4]) {
                /* Write C to file */
                writeColMajorMatrixFile(argv[4], matArow, matBcol, matC);
        }

        return 0;
}
