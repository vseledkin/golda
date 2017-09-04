package main

import (
	"log"
	"flag"
	"fmt"
	"os"
	".."
)

const (
	model = "model"
)

var input, cwd string
var t int // number of topics

func main() {
	modelCommand := flag.NewFlagSet(model, flag.ExitOnError)
	modelCommand.StringVar(&input, "input", "", "text file one document per line")
	modelCommand.IntVar(&t, "t", 2, "number of topics")

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
		if t < 2 {
			modelCommand.PrintDefaults()
		}
		err := Model(t, input)
		if err != nil {
			panic(err)
		}
		return
	}
}

func Model(t int, input string) error {
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

	lda := new(golda.LDA).Init(t, corpus, dictionary)

	{
		numel, mb := lda.WT.MSize()
		log.Printf("Model WT size %d %f MB\n", numel, mb)
	}

	log.Printf("%#v\n", corpus[10])

	for i := 0; i < 40; i++ {
		lda.Step()
		log.Printf("Step %d\n", i)
		lda.PrintTopics(5)
	}

	return nil
}
