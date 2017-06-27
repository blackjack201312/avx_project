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

#include <iostream>


void basicSgemm( char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc )
{
        if ((transa != 'T') && (transa != 't')) {
                std::cerr << "unsupported value of 'transa' in regtileSgemm()" << std::endl;
                return;
        }

        if ((transb != 'N') && (transb != 'n')) {
                std::cerr << "unsupported value of 'transb' in regtileSgemm()" << std::endl;
                return;
        }

//#pragma omp parallel for collapse (2)
//        for (int mm = 0; mm < m; ++mm) {
//                for (int nn = 0; nn < n; ++nn) {
//                        float c = 0.0f;
//                        for (int i = 0; i < k; ++i) {
//                                float a = A[mm + i * lda];
//                                float b = B[nn + i * ldb];
//                                c += a * b;
//                        }
//                        C[mm+nn*ldc] = C[mm+nn*ldc] * beta + alpha * c;
//                }
//        }





  #pragma omp parallel for collapse (2)
  for (int mm = 0; mm < m; ++mm) {
    for (int nn = 0; nn < n; ++nn) {
      float c = 0.0f;
      #pragma omp simd  
      for (int i = 0; i < k; ++i) {
        // float a = A[mm * k + i]; 
        // float b = B[nn * k + i];
        c += A[mm * k + i] * B[nn * k + i];
      }
      C[mm+nn*ldc] = C[mm+nn*ldc] * beta + alpha * c;
    }
  }
}
