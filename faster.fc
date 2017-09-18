#include "faster.h"

void* MakeAlignedSlice(int alignment, int size, int length){
	void* memory;
	posix_memalign(&memory, alignment, length * size);
	return memory;
}

void updateTheta(int d, int J, float* theta, float* mu, uint32_t* jc, float* sr) {
	int dstop = jc[d+1];
	float srv;
	float* thetap = theta + d*J;
	float* mup = mu + jc[d]*J;
	int i, j;
	//printf("Document %d\n", d);
	for (i = jc[d]; i < dstop; i++) {
		//printf("Word %d\n", i);
		axpy_sse(sr[i],mup,thetap,J);
		//srv = sr[i];
		//for (j = 0; j < J; j++) {
		//	thetap[j] += srv*mup[j]; // increment theta count matrix
		//}

		mup += J;
	}
}

void passMessage(float ALPHA,float WBETA,float BETA
,int d , int J, float* phi, float* phitot, float* theta,float* mu
, uint32_t* jc, uint32_t* ir,float* sr) {
	/* passing message mu */
	int i, j, stop = jc[d + 1];
	float srv, delta;//, mutot;

	float* thetap = theta + d*J;
	float* mup = mu + jc[d]*J;
	float* phip;
	for (i=jc[d]; i<stop; ++i) {
		phip = phi + ir[i]*J;
		word_message_sse(BETA, WBETA, ALPHA, sr[i], mup, phip, phitot, thetap,J);
		//for (j =0;j<J;j++){
		//	delta = sr[i]*mup[j];
		//	mup[j] = (phip[j] - delta + BETA) / (phitot[j] - delta + WBETA) * (thetap[j] - delta + ALPHA);
		//}
		scale_sse(1/sum_sse(mup,J), mup, J);
		mup += J;
	}
}

void word_message_sse(float BETA, float WBETA, float ALPHA, float SRV, float* mu, float *phi, float* phitot, float* theta, int l){
	const __m128 srv = _mm_set_ps1( SRV );
	const __m128 beta = _mm_set_ps1( BETA );
	const __m128 wbeta = _mm_set_ps1( WBETA );
	const __m128 alpha = _mm_set_ps1( ALPHA );
	__m128* mup = (__m128*) mu;
	__m128* phip = (__m128*) phi;
	__m128* phitotp = (__m128*) phitot;
	__m128* thetap = (__m128*) theta;
	__m128 delta, phipart, totpart, thetapart;
	float delt;

	int i, I = l / 4;
	//printf("%d %d %d %d\n",mu, l, I, 4*I);
	for (i = 0; i < I; ++i, ++mup, ++phip, ++phitotp, ++thetap) {
		//printf("strade->%d\n",i);
		delta = _mm_mul_ps( srv, *mup );
		phipart = _mm_add_ps(_mm_sub_ps(*phip, delta), beta);
		totpart = _mm_add_ps(_mm_sub_ps(*phitotp, delta), wbeta);
		thetapart = _mm_add_ps(_mm_sub_ps(*thetap, delta), alpha);
		*mup = _mm_div_ps(_mm_mul_ps(phipart,thetapart), totpart);
		//printf("afterstrade->%d\n",i);
	}

	for (i = 4*I; i < l; ++i) {
		//printf("REMAINS->%d %d %d\n",i, I, 4*I);
		delt = SRV*mu[i];
		mu[i] = (phi[i] - delt + BETA) / (phitot[i] - delt + WBETA) * (theta[i] - delt + ALPHA);
	}
	//printf("END->%d %d %d\n",l, I, 4*I);
}

void axpy_sse(float a, float* x , float* y, int l){
	const __m128 aaaa = _mm_set_ps1( a );
	__m128* X = (__m128*) x;
	__m128* Y = (__m128*) y;
	int i, I = l / 4;
	//printf("%d %d %d %d\n",x, l, I, 4*I);
	for (i = 0; i < I; ++i, ++X, ++Y) {
		//printf("strade->%d\n",i);
		*Y = _mm_add_ps(_mm_mul_ps(aaaa, *X), *Y);
		//printf("afterstrade->%d\n",i);
	//	x+=4;
		//y+=4;
	}
	for (i = 4*I; i < l; ++i) {
		y[i] += a*x[i];
	}
	//printf("->%d %d %d\n",l, I, 4*I);
}

void clean_sse(float* x, int l){
  __m128* ptr = (__m128*) x;
	int i, I = l / 4;
  for (i = 0; i < I; ++i, ++ptr) {
		*ptr = _mm_setzero_ps();
	}
	for (i = 4*I; i < l; ++i) {
		x[i] = .0;
	};
}

void scale_sse(float a, float* x, int l){
		// We assume N % 4 == 0.
	const __m128 aaaa = _mm_set_ps1( a );
	__m128* ptr = (__m128*) x;
	int i, I = l / 4;
	for (i = 0; i < I; ++i, ++ptr) {
		*ptr = _mm_mul_ps( aaaa, *ptr );
	}
	for (i = 4*I; i < l; ++i) {
		x[i] *= a;
	};
}

float sum_sse(float* x, int l){
		// We assume N % 4 == 0.
	__m128 ssss = _mm_set_ps1( .0 );
	__m128* ptr = (__m128*) x;
	int i, I = l / 4;
	for (i = 0; i < I; ++i, ++ptr) {
		ssss = _mm_add_ps( ssss, *ptr );
	}
	float s = ssss[0] + ssss[1] +ssss[2] +ssss[3];
	for (i = 4*I; i < l; ++i) {
		s+= x[i];
	}
	return s;
}
