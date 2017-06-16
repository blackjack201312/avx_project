/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/
#include <omp.h>
#define PI   3.1415926535897932384626433832795029f
#define PIx2 6.2831853071795864769252867665590058f

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define K_ELEMS_PER_GRID 2048

struct kValues {
  float Kx;
  float Ky;
  float Kz;
  float PhiMag;
};

#pragma offload_attribute(push, target(mic))
inline
void 
ComputePhiMagCPU(int numK, 
                 float* phiR, float* phiI, float* phiMag) {
  int indexK = 0;
  #pragma omp parallel for 
  for (indexK = 0; indexK < numK; indexK++) {
    float real = phiR[indexK];
    float imag = phiI[indexK];
    phiMag[indexK] = real*real + imag*imag;
  }
}

inline
void
ComputeQCPU(int numK, int numX,
//            struct kValues *kVals,
	    float* kx, float* ky, float* kz, float* phiMag,
            float* x, float* y, float* z,
            float *Qr, float *Qi) {
  float expArg;
  float cosArg;
  float sinArg;

  int indexK, indexX;
  #pragma omp parallel for private(indexK,expArg,cosArg,sinArg) 
  for (indexX = 0; indexX < numX; indexX++) {
    for (indexK = 0; indexK < numK; indexK++) {
//      expArg = PIx2 * (kVals[indexK].Kx * x[indexX] +
//                       kVals[indexK].Ky * y[indexX] +
//                       kVals[indexK].Kz * z[indexX]);
      expArg = PIx2 * (kx[indexK] * x[indexX] +
	               ky[indexK] * y[indexX] + 
		       kz[indexK] * z[indexX]);

      cosArg = cosf(expArg);
      sinArg = sinf(expArg);

      float phi = phiMag[indexK];

      Qr[indexX] += phi * cosArg;
      Qi[indexX] += phi * sinArg;
    }
  }
}
#pragma offload_attribute(pop)

void createDataStructsCPU(int numK, int numX, float** phiMag,
	 float** Qr, float** Qi)
{
  *phiMag = (float* ) memalign(16, numK * sizeof(float));
  *Qr = (float*) memalign(16, numX * sizeof (float));
  memset((void *)*Qr, 0, numX * sizeof(float));
  *Qi = (float*) memalign(16, numX * sizeof (float));
  memset((void *)*Qi, 0, numX * sizeof(float));
}
