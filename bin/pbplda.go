package main

import (
	"flag"
	"fmt"
	"runtime"
	"time"
	"unsafe"
)

func main() {
	v := make([]float32, 100)
	fmt.Println("[]float32 align ", unsafe.Alignof(v))
	v64 := make([]float64, 100)
	fmt.Println("[]float64 align ", unsafe.Alignof(v64))
	runtime.GOMAXPROCS(runtime.NumCPU())
	fmt.Println("CPU:", runtime.NumCPU())

	var in, out, voc string
	var iterations, topics, threads, alignment int
	flag.StringVar(&in, "i", "text.mat", "Matrix file path to load")
	flag.StringVar(&voc, "v", "vocabulary.voc", "Vocabulary file path to load from")
	flag.StringVar(&out, "phi", "text.phi", "Phi file path to save")
	flag.IntVar(&iterations, "m", 100, "Maximum number of iterations to process")
	flag.IntVar(&threads, "t", 4, "Maximum number of parallel threads to run")
	flag.IntVar(&topics, "j", 10, "Number of topics")
	flag.IntVar(&alignment, "a", 8, "Data alignment")
	flag.Parse()
	// load matrix
	start := time.Now()
	m := ai.LoadMatrix(in)
	fmt.Printf("Loaded data matrix (%d,%d) of %d nonzeros for %v\n", m.Rows, m.Cols, m.Nonzeros, time.Now().Sub(start))
	// run APBP
	start = time.Now()
	var (
		ALPHA     float32 = 0.01
		BETA      float32 = 0.01
		OUTPUT            = true
		startcond         = false
	)

	mu := make([]float32, topics*int(m.Nonzeros))
	//mu := ai.MakeAlignedSliceOfFloat32(alignment, topics*int(m.Nonzeros))
	phi := make([]float32, topics*int(m.Rows))
	//phi := ai.MakeAlignedSliceOfFloat32(alignment, topics*int(m.Rows))
	theta := make([]float32, topics*int(m.Cols))
	//theta := ai.MakeAlignedSliceOfFloat32(alignment, topics*int(m.Cols))
	// start profiler
	//profilef, err := os.Create("cpu.pbplda.profile")
	//if err != nil {
	//	log.Fatal(err)
	//}

	//pprof.StartCPUProfile(profilef)
	//m.SyncronousParallelBeliefPropagation(ALPHA, BETA, topics, iterations, startcond, OUTPUT, phi, theta, mu, threads, alignment)
	//ALPHA, BETA float32, J, NN int, startcond, OUTPUT bool, phi, theta, mu []float32
	//m.AsyncronousParallelBeliefPropagation(ALPHA, BETA, topics, iterations, startcond, OUTPUT, phi, theta, mu)
	m.AsyncronousBeliefPropagation(ALPHA, BETA, uint32(topics), uint32(iterations), startcond, OUTPUT, phi, theta, mu)
	//pprof.StopCPUProfile()
	ai.SaveFloat32Matrix("phi.matrix", phi, m.Rows, uint32(topics))

	pldadur := time.Now().Sub(start)
	start = time.Now()
	//dictionary := ai.LoadDictionary(voc)
	dictionary := aio.Load(voc, ai.Dictionary{}).(*ai.Dictionary)
	dictionary.WordByID(0)

	ai.PrintTopics(dictionary, phi, uint32(topics), m.Rows, 10)
	fmt.Printf("Printiing finished for %v\n", time.Now().Sub(start))
	fmt.Printf("PBPLDA finished for %v\n", pldadur)

}
