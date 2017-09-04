package golda

import (
	"os"
	"bufio"
)

type Matrix struct {
	W [][]float32
}

func (m Matrix) MSize() (int, float32) {
	r := len(m.W)
	if r > 0 {
		c := len(m.W[0])
		return r * c, float32(r*c*4) / 1024 / 1024
	}
	return 0, 0
}

func (m Matrix) Resize(rows, columns int) Matrix {
	m.W = make([][]float32, rows)
	for i := range m.W {
		m.W[i] = make([]float32, columns)
	}
	return m
}

type Document struct {
	text   []string
	tokens []int
	ta []int
}

type Corpus []*Document

func (c Corpus) Len() int {
	return len(c)
}


func LoadCorpus(filename string) (corpus Corpus, dic *Dictionary, err error) {
	dic, err = DictionaryFromFile(filename)
	if err != nil {
		return
	}
	file, err := os.Open(filename)
	if err != nil {
		return
	}
	defer file.Close()

	r := bufio.NewReader(file)
	for {
		line, e := r.ReadString('\n')
		if e != nil {
			break
		}
		if len(line) > 0 {
			tokens := Split(line)
			corpus = append(corpus, &Document{tokens, dic.Tokens2Digits(tokens), nil})
		}
	}
	return
}
