package golda

/*
SparseMatrix compressed sparse matrix
*/
type SparseMatrix struct {
	index, Nonzeros, Rows, Cols uint32
	Ir, Jc                      []uint32
	Sr                          []float32
}

/*
NewSparseMatrix allocates new matrix
*/
func NewSparseMatrix(nonzeros, rows, cols uint32) (matrix *SparseMatrix) {
	matrix = &SparseMatrix{Nonzeros: nonzeros, Rows: rows, Cols: cols}
	matrix.Ir = make([]uint32, nonzeros)
	matrix.Jc = make([]uint32, cols+1)
	matrix.Sr = make([]float32, nonzeros)
	return
}

/*
Set sets value v of  the [i,j] cell
*/
func (m *SparseMatrix) Set(i, j uint32, v float32) {
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
	var j uint32
	for ; j < m.Cols; j++ {
		m.Jc[j+1] += m.Jc[j]
	}
	m.Jc[0] = 0
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
