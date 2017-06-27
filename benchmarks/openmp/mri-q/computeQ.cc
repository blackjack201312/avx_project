#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <malloc.h>

#define PI   3.1415926535897932384626433832795029f
#define PIx2 6.2831853071795864769252867665590058f

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define K_ELEMS_PER_GRID 2048

//struct kValues {
//	float Kx;
//	float Ky;
//	float Kz;
//	float PhiMag;
//};

inline
void
ComputePhiMagCPU(int numK,
                float* phiR, float* phiI, float* phiMag) {
        float real, imag;
        int indexK = 0;
#pragma omp parallel for simd    //The simd here make little sense.
        for (indexK = 0; indexK < numK; indexK++) {
                real = phiR[indexK];
                imag = phiI[indexK];
                phiMag[indexK] = real*real + imag*imag;
        }
}

inline
void
ComputeQCPU(int numK, int numX,
                float *Kx, float *Ky, float *Kz, float *PhiMag,
                float* x, float* y, float* z,
                float *Qr, float *Qi) {
        float expArg;
        float cosArg;
        float sinArg;

        int indexK, indexX;
#pragma omp parallel for
        for (indexK = 0; indexK < numK; indexK++) {
#pragma omp simd //The simd instruction here matters!!!(Half time)
                for (indexX = 0; indexX < numX; indexX++) {
                        expArg = PIx2 * (Kx[indexK] * x[indexX] +
                                        Ky[indexK] * y[indexX] +
                                        Kz[indexK] * z[indexX]);

                        cosArg = cosf(expArg);
                        sinArg = sinf(expArg);

                        float phi = PhiMag[indexK];
                        Qr[indexX] += PhiMag[indexK] * cosArg;
                        Qi[indexX] += PhiMag[indexK] * sinArg;
                }
        }
}

void createDataStructsCPU(int numK, int numX, float** phiMag,
                float** Qr, float** Qi)
{
        *phiMag = (float* ) memalign(16, numK * sizeof(float));
        *Qr = (float*) memalign(16, numX * sizeof (float));
        memset((void *)*Qr, 0, numX * sizeof(float));
        *Qi = (float*) memalign(16, numX * sizeof (float));
        memset((void *)*Qi, 0, numX * sizeof(float));
}
