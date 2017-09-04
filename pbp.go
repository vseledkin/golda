package golda

/*
#include "faster.h"
*/
import "C"

import (
	"../asm"
	"fmt"
	"math"
	"math/rand"
	"time"
)

/*
AsyncronousBeliefPropagation - runs lda
*/
func (m *SparseMatrix) AsyncronousBeliefPropagation(ALPHA, BETA float32, J, NN uint32, startcond, OUTPUT bool, phi, theta, mu []float32) {
	rand.Seed(time.Now().UnixNano())
	var (
		W                      = m.Rows
		D                      = m.Cols
		wi, di, j, topic, iter uint32
		i                      uint32
		xitot, mutot, xi       float32
		perp                   float64
	)
	WBETA := float32(W) * BETA
	JALPHA := float32(J) * ALPHA
	phitot := make([]float32, J)
	thetad := make([]float32, D)
	jc := m.Jc
	ir := m.Ir
	sr := m.Sr
	/* random initialization */
	for di = 0; di < D; di++ { // go throught documents
		for i = jc[di]; i < jc[di+1]; i++ { // go throught columns
			wi = ir[i]
			xi = sr[i]
			thetad[di] += xi
			xitot += xi
			// pick a random topic 0..J-1
			topic = uint32(rand.Intn(int(J)))
			mu[i*J+topic] = 1.0     // assign this word token to this topic
			phi[wi*J+topic] += xi   // increment phi count matrix
			theta[di*J+topic] += xi // increment theta count matrix
			phitot[topic] += xi     // increment phitot matrix
		}
	}

	for iter = 0; iter < NN; iter++ {
		if OUTPUT {
			if (iter%10) == 0 && (iter != 0) {
				/* calculate perplexity */
				perp = 0.0
				for di = 0; di < D; di++ {
					for i = jc[di]; i < jc[di+1]; i++ {
						wi = ir[i]
						xi = sr[i]
						mutot = 0.0
						for j = 0; j < J; j++ {
							mutot += (phi[wi*J+j] + BETA) /
								(phitot[j] + WBETA) *
								(theta[di*J+j] + ALPHA) /
								(thetad[di] + JALPHA)
						}
						perp -= math.Log(float64(mutot)) * float64(xi)
					}
				}
				if iter % 100 == 0 {
					fmt.Printf("\tIteration %d of %d:\t%0.2f\n", iter, NN, math.Exp(perp/float64(xitot)))
				}
			}
		}

		var iJj uint32
		for di = 0; di < D; di++ {
			for i = jc[di]; i < jc[di+1]; i++ {
				wi = ir[i]
				xi = sr[i]
				mutot = 0
				for j = 0; j < J; j++ {
					iJj = i*J + j
					phi[wi*J+j] -= xi * mu[iJj]
					phitot[j] -= xi * mu[iJj]
					theta[di*J+j] -= xi * mu[iJj]
					mu[iJj] = (phi[wi*J+j] + BETA) / (phitot[j] + WBETA) * (theta[di*J+j] + ALPHA)
					mutot += mu[iJj]
				}
				for j = 0; j < J; j++ {
					mu[i*J+j] /= mutot
					phi[wi*J+j] += xi * mu[i*J+j]
					phitot[j] += xi * mu[i*J+j]
					theta[di*J+j] += xi * mu[i*J+j]
				}
			}
		}
	}
}

func zeron(n uint32, v float32) (z []float32) {
	z = make([]float32, n)
	for i := range z {
		z[i] = v
	}
	return
}

/*
EvaluatePerplexity evaluates topic model quality in terms of perplexity
*/
func (m *SparseMatrix) EvaluatePerplexity(ALPHA, BETA, xitot float32, phi, theta, phitot, thetad []float32, drange, jrange []struct{}) (perp float64) {
	/* calculate perplexity */
	var (
		W            = m.Rows
		di, j        int
		i            uint32
		mutot        float32
		thetap, phip []float32
		J            = len(jrange)
	)
	WBETA := float32(W) * BETA
	JALPHA := float32(J) * ALPHA
	jc := m.Jc
	ir := m.Ir
	sr := m.Sr
	for di = range drange {
		thetap = theta[di*J : di*J+J]
		for i = jc[di]; i < jc[di+1]; i++ {
			mutot = 0.0
			phip = phi[int(ir[i])*J : int(ir[i])*J+J]
			for j = range jrange {
				mutot += (phip[j] + BETA) /
					(phitot[j] + WBETA) *
					(thetap[j] + ALPHA) /
					(thetad[di] + JALPHA)
			}
			perp -= math.Log(float64(mutot)) * float64(sr[i])
		}
	}
	perp = math.Exp(perp / float64(xitot))
	return
}

/*
AsyncronousParallelBeliefPropagation - runs lda it is more presize but
very jard to parallelize
*/
func (m *SparseMatrix) AsyncronousParallelBeliefPropagation(ALPHA, BETA float32, J, NN int, startcond, OUTPUT bool, phi, theta, mu []float32) {
	rand.Seed(time.Now().UnixNano())
	var (
		W                        = m.Rows
		D                        = int(m.Cols)
		wi, d, i, j, topic, iter int
		xitot, xi, mutot         float32
		mup, phip, thetap        []float32
		drange                   = make([]struct{}, D)
		jrange                   = make([]struct{}, J)
	)
	//BETAVECTOR := zeron(J, BETA)
	//ALPHAVECTOR := zeron(J, ALPHA)
	WBETA := float32(W) * BETA
	//WBETAVECTOR := zeron(J, float32(W)*BETA)
	phitot := make([]float32, J)
	thetad := make([]float32, D)
	jc := m.Jc
	ir := m.Ir
	sr := m.Sr
	/* random initialization */
	for d = 0; d < int(m.Cols); d++ { // go throught documents
		for i = int(jc[d]); i < int(jc[d+1]); i++ { // go throught columns
			wi = int(ir[i])
			xi = sr[i]
			thetad[d] += xi
			xitot += xi
			// pick a random topic 0..J-1
			topic = rand.Intn(J)
			mu[i*J+topic] = 1.0    // assign this word token to this topic
			phi[wi*J+topic] += xi  // increment phi count matrix
			theta[d*J+topic] += xi // increment theta count matrix
			phitot[topic] += xi    // increment phitot matrix
		}
	}
	start := time.Now()
	for iter = 0; iter < NN; iter++ {
		if OUTPUT {
			if (iter%10) == 0 && (iter != 0) {
				go func() {
					perp := m.EvaluatePerplexity(ALPHA, BETA, xitot, phi, theta, phitot, thetad, drange, jrange)
					fmt.Printf("\tIteration %d of %d:\t%0.2f %v\n", iter, NN, perp, time.Now().Sub(start))
					start = time.Now()
				}()
			}
		}

		//var threads = 2
		//sync := make(chan empty, threads)
		//for t := 0; t < threads; t++ {
		//	sync <- empty{}
		//}
		for d = 0; d < D; d++ {
			//<-sync
			//go func(d uint32) {
			for i = int(jc[d]); i < int(jc[d+1]); i++ {
				mutot = 0
				mup = mu[i*J : i*J+J]
				phip = phi[int(ir[i])*J : int(ir[i])*J+J]
				thetap = theta[d*J : d*J+J]
				asm.Saxpy(-sr[i], mup, phip)
				asm.Saxpy(-sr[i], mup, phitot)
				asm.Saxpy(-sr[i], mup, thetap)

				// vectorize gradient
				//blas.Scopy(J, BETAVECTOR, 1, mup, 1)
				//blas.Saxpy(J, 1, phi[ir[i]*J:ir[i]*J+J], 1, mup, 1)
				for j = 0; j < J; j++ {
					mup[j] = (phip[j] + BETA) / (phitot[j] + WBETA) * (thetap[j] + ALPHA)
				}
				mutot = asm.Ssum(mup)

				asm.Sscale(1/mutot, mup)
				asm.Saxpy(sr[i], mup, phip)
				asm.Saxpy(sr[i], mup, phitot)
				asm.Saxpy(sr[i], mup, thetap)
			}
			//sync <- empty{}
			//}(did)
		}
		//for t := 0; t < threads; t++ {
		//	<-sync
		//}
	}
}

/*
SyncronousParallelBeliefPropagation - runs lda it is more presize but
very jard to parallelize
*/
func (m *SparseMatrix) SyncronousParallelBeliefPropagation(ALPHA, BETA float32, J, NN int, startcond, OUTPUT bool, phi, theta, mu []float32, threads, alignment int) {
	rand.Seed(time.Now().UnixNano())
	var (
		W                        = m.Rows
		D                        = int(m.Cols)
		i, topic, iter, d, dstop int
		xitot                    float32
		//mup, thetap                      []float32
		drange = make([]struct{}, D)
		jrange = make([]struct{}, J)
	)
	WBETA := float32(W) * BETA
	phitot := make([]float32, J)
	thetad := make([]float32, D)
	jc := m.Jc
	ir := m.Ir
	sr := m.Sr

	/* random initialization */
	for d = range drange { // go throught documents
		for i = int(jc[d]); i < int(jc[d+1]); i++ { // go throught columns
			thetad[d] += sr[i]
			xitot += sr[i]
			// pick a random topic 0..J-1
			topic = rand.Intn(J)
			mu[i*J+topic] = 1.0              // assign this word token to this topic
			phi[int(ir[i])*J+topic] += sr[i] // increment phi count matrix
			theta[d*J+topic] += sr[i]        // increment theta count matrix
			phitot[topic] += sr[i]           // increment phitot matrix
		}
	}
	start := time.Now()
	//sync := makeSync(threads)
	sync := make(chan interface{}, threads)
	for t := 0; t < threads; t++ {
		sync <- struct{}{}
	}
		
	for iter = range make([]struct{}, NN) {
		//go func() {
		if OUTPUT {
			if (iter%10) == 0 && (iter != 0) {
				perp := m.EvaluatePerplexity(ALPHA, BETA, xitot, phi, theta, phitot, thetad, drange, jrange)
				fmt.Printf("\tIteration %d of %d:\t%0.2f %v\n", iter, NN, perp, time.Now().Sub(start))
				start = time.Now()
			}
		}
		//}()

		/* passing message mu */
		for d = range drange {
			<-sync
			go func(d int) {
				C.passMessage(C.float(ALPHA), C.float(WBETA), C.float(BETA), C.int(d), C.int(J),
					(*C.float)(&phi[0]), (*C.float)(&phitot[0]), (*C.float)(&theta[0]), (*C.float)(&mu[0]),
					(*C.uint32_t)(&m.Jc[0]), (*C.uint32_t)(&m.Ir[0]), (*C.float)(&m.Sr[0]))
				sync <- struct{}{}
			}(d)
		}
		
		for t := 0; t < threads; t++ {
			<-sync
		}
		/* clear phi, theta, and phitot */
		asm.Sclean(phi)
		asm.Sclean(theta)
		asm.Sclean(phitot)

		
		/* update theta, phi and phitot using message mu */
		for d = range drange {
			<-sync
			go func(d int) {
				C.updateTheta(C.int(d), C.int(J), (*C.float)(&theta[0]), (*C.float)(&mu[0]), (*C.uint32_t)(&m.Jc[0]), (*C.float)(&m.Sr[0]))
				sync <- struct{}{}
			}(d)

			//go m.UpdateΘ(d, J, theta, mu, sync)
			//<-sync
			//go func(d int) {
			m.UpdateΦ(d, J, phi, mu, nil)
			dstop = int(jc[d+1])
			for i = int(jc[d]); i < dstop; i++ {
				asm.Saxpy(sr[i], mu[i*J:i*J+J], phitot)
			}
			//sync <- struct{}{}
			//}(d)
		}
		for t := 0; t < threads; t++ {
			<-sync
		}
	}
}

/*
UpdateΘ - aupdate theta matrix
*/
func (m *SparseMatrix) UpdateΘ(d, J int, Θ, µ []float32, sync chan struct{}) {
	dstop := int(m.Jc[d+1])
	for i := int(m.Jc[d]); i < dstop; i++ {
		asm.Saxpy(m.Sr[i], µ[i*J:i*J+J], Θ[d*J:d*J+J])
	}
	sync <- struct{}{}
}

/*
UpdateΦ - aupdate Φ matrix
*/
func (m *SparseMatrix) UpdateΦ(d, J int, Φ, µ []float32, sync chan struct{}) {
	dstop := int(m.Jc[d+1])
	for i := int(m.Jc[d]); i < dstop; i++ {
		asm.Saxpy(m.Sr[i], µ[i*J:i*J+J], Φ[int(m.Ir[i])*J:int(m.Ir[i])*J+J])
	}
	if sync != nil {
		sync <- struct{}{}
	}
}
