package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	"bufio"

	".."
)

const (
	model = "model"
)

var input, cwd string
var topics int // number of topics

func main() {
	modelCommand := flag.NewFlagSet(model, flag.ExitOnError)
	modelCommand.StringVar(&input, "input", "", "text file one document per line")
	modelCommand.IntVar(&topics, "t", 2, "number of topics")

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage of %s:\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "%s <command> arguments\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "commands are:\n")

		fmt.Fprintf(os.Stderr, "%s\n", model)
		modelCommand.PrintDefaults()

		flag.PrintDefaults()
	}
	flag.Parse()
	log.SetOutput(os.Stderr)

	if len(os.Args) == 1 {
		flag.Usage()
		os.Exit(1)
	}

	cwd, _ = os.Getwd()
	log.Printf("Starting in %s directory.", cwd)
	switch os.Args[1] {
	case model:
		modelCommand.Parse(os.Args[2:])
	default:
		log.Printf("%q is not valid command.\n", os.Args[1])
		os.Exit(1)
	}

	// MODEL COMMAND ISSUED
	if modelCommand.Parsed() {
		if input == "" {
			modelCommand.PrintDefaults()
			return
		}
		if topics < 2 {
			modelCommand.PrintDefaults()
		}
		err := Model(topics, input)
		if err != nil {
			panic(err)
		}
		return
	}
}

func tf(t string) map[string]int {
	m := make(map[string]int)
	for _, token := range golda.Split(t) {
		m[token]++
	}
	return m
}

func Model(t int, input string) error {
	dic, err := golda.DictionaryFromFile(input)
	if err != nil {
		return err
	}
	i := 0
	for i < 10 {
		t := dic.TokenByID(i)
		log.Printf("%d %s %d\n", t.ID, t.Token, t.Frequency)
		i++
	}
	log.Printf("Dictionary has %d tokens\n", dic.Len())

	// count number of nonzeros
	nnz := 0
	documents := 0
	f, err := os.Open(input)
	if err != nil {
		return err
	}
	r := bufio.NewReader(f)
	for {
		line, e := r.ReadString('\n')
		if e != nil {
			break
		}

		if len(line) > 0 {
			for token, count := range tf(line) {
				if _, ok := dic.Tokens[token]; ok {
					nnz += count
				}
			}
			documents++
		}
	}
	m := golda.NewSparseMatrix(nnz, dic.Len(), documents)
	f.Seek(0, 0)
	documents = 0
	for {
		line, e := r.ReadString('\n')
		if e != nil {
			break
		}

		if len(line) > 0 {
			for token, count := range tf(line) {
				if t, ok := dic.Tokens[token]; ok {
					m.Set(t.ID, documents, float32(count))
				}
			}
			documents++
		}
	}
	f.Close()
	m.Pack()
	// make sparce matrix
	log.Printf("Corpus has %d documents\n", documents)
	log.Printf("Matrix has %d rows\n", m.Rows)
	log.Printf("Matrix has %d columns\n", m.Cols)
	log.Printf("Matrix has %d nonzeros\n", m.Nonzeros)

	sparsity := 100 * float64(nnz) / (float64(m.Rows) * float64(m.Cols))
	fmt.Printf("Matrix sparsity: %3.3f %%\n", sparsity)

	var (
		ALPHA     float32 = 0.01
		BETA      float32 = 0.1
		OUTPUT            = true
		startcond         = false
	)
	mu := make([]float32, topics*int(m.Nonzeros))
	phi := make([]float32, topics*int(m.Rows))
	theta := make([]float32, topics*int(m.Cols))
	m.AsyncronousBeliefPropagation(ALPHA, BETA, topics, 10000, startcond, OUTPUT, phi, theta, mu)
	PrintTopics(dic, phi, topics, m.Rows, 10)
	return nil
}

/*
WeightedUint32 unsigned weighted number
*/
type WeightedInt struct {
	ID     int
	Weight float32
}

/*
PrintTopics -prints top N words of every topic
*/
func PrintTopics(d *golda.Dictionary, phi []float32, topics, words, n int) {
	var weight float32
	for t := 0; t < topics; t++ {
		fmt.Printf("Topic %d:\n", t)
		items := make([]WeightedInt, n)
		for i := 0; i < words; i++ {
			weight = phi[i*topics+t]
			for j := 0; j < n; j++ {
				if weight > items[j].Weight {
					for p := n - 1; p > j; p-- {
						items[p].Weight = items[p-1].Weight
					}
					items[j].ID = i
					items[j].Weight = weight
					break
				}
			}
		}
		for i := 0; i < n; i++ {
			fmt.Printf("\t%d %d %f %s\n", i, items[i].ID, items[i].Weight, d.TokenByID(items[i].ID).Token)
		}
	}
}

/*
func maisn() {
	corpus, dictionary, err := golda.LoadCorpus(input)
	if err != nil {
		return err
	}
	i := 0
	for i < 10 {
		t := dictionary.TokenByID(i)
		log.Printf("%d %s %d\n", t.ID, t.Token, t.Frequency)
		i++
	}
	log.Printf("Dictionary has %d tokens\n", dictionary.Len())
	log.Printf("Corpus has %d documents\n", corpus.Len())

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
*/
