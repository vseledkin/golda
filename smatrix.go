package golda

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

/*
SparseMatrix compressed sparse matrix
*/
type SparseMatrix struct {
	index, Nonzeros, Rows, Cols int
	Ir, Jc                      []int
	Sr                          []float32
}

/*
NewSparseMatrix allocates new matrix
*/
func NewSparseMatrix(nonzeros, rows, cols int) (matrix *SparseMatrix) {
	matrix = &SparseMatrix{Nonzeros: nonzeros, Rows: rows, Cols: cols}
	matrix.Ir = make([]int, nonzeros)
	matrix.Jc = make([]int, cols+1)
	matrix.Sr = make([]float32, nonzeros)
	return
}

/*
Set sets value v of  the [i,j] cell
*/
func (m *SparseMatrix) Set(i, j int, v float32) {
	if m.index < m.Nonzeros {
		m.Ir[m.index] = i
		m.Jc[j]++
		m.Sr[m.index] = v
		m.index++
	} else {
		panic("Sparse matrix overfill")
	}
}

/*
Pack packs matrix
*/
func (m *SparseMatrix) Pack() {
	for j := 0; j < m.Cols; j++ {
		m.Jc[j+1] += m.Jc[j]
	}
	m.Jc[0] = 0
}

/*
AsyncronousBeliefPropagation - runs lda
*/
func (m *SparseMatrix) AsyncronousBeliefPropagation(ALPHA, BETA float32, J, NN int, startcond, OUTPUT bool, phi, theta, mu []float32) {
	rand.Seed(time.Now().UnixNano())
	var (
		W                      = m.Rows
		D                      = m.Cols
		wi, di, j, topic, iter int
		i                      int
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
			topic = rand.Intn(J)
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
				if iter%100 == 0 {
					fmt.Printf("\tIteration %d of %d:\t%0.2f\n", iter, NN, math.Exp(perp/float64(xitot)))
				}
			}
		}

		var iJj int
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

/*
EvaluatePerplexity evaluates topic model quality in terms of perplexity
*/
func (m *SparseMatrix) EvaluatePerplexity(ALPHA, BETA, xitot float32, phi, theta, phitot, thetad []float32, drange, jrange []struct{}) (perp float64) {
	/* calculate perplexity */
	var (
		W            = m.Rows
		di, j, i     int
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
LoadMatrix loads sparse matrix from file
*/
/*
func LoadMatrix(infile string) (m *SparseMatrix) {
	unzipCloser, unzip, e := aizip.NewCompressedReader(infile)
	if e != nil {
		panic(e)
	}
	nonzeros := unzipCloser.Data2
	rows := unzipCloser.Data0
	columns := unzipCloser.Data1
	sparsity := 100 * float64(nonzeros) / (float64(rows) * float64(columns))
	fmt.Println("Loading matrix Rows:", rows, "Cols:", columns, "NNZ:", nonzeros, " sparsity:", sparsity, "%")
	m = NewSparseMatrix(nonzeros, rows, columns)
	var bow BowDocument

	for e := (&bow).ReadFill(unzip); e == nil; e = (&bow).ReadFill(unzip) {
		//fmt.Println(bow)
		for i := range bow.WordID {
			m.Set(bow.WordID[i], bow.ID, bow.DF[i]) // can be highly optimized via copy
		}
	}
	unzipCloser.Close()
	m.Pack()
	return
}*/
