#include <stdlib.h>
#include <stdint.h>
//#include <xmmintrin.h>
#include <inttypes.h>
#include <immintrin.h>

void* MakeAlignedSlice(int alignment, int size, int length);

void updateTheta(int d, int J, float* theta, float* mu, uint32_t* jc, float* sr);

void passMessage(float ALPHA,float WBETA,float BETA
,int d, int J, float* phi, float* phitot, float* theta,float* mu
,uint32_t* jc,uint32_t* ir,float* sr);

void word_message_sse(float BETA, float WBETA, float ALPHA, float SRV, float* mu, float *phi, float* phitot, float* theta, int l);

void axpy_sse(float a, float* x , float* y, int l);

void clean_sse(float* x, int l);

void scale_sse(float a, float* x, int l);

float sum_sse(float* x, int l);
