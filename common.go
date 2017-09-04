package golda

import (
	"fmt"
	"unicode"
	"strings"
	"math/rand"
)

func Range(i int) []struct{} {
	return make([]struct{}, i)
}

func Multinomial(probabilities []float32) int {
	offset := float32(0)
	sample := rand.Float32()
	for i, p := range probabilities {
		offset += p
		//sample uniform from [0,1]
		if sample <= offset {
			return i
		}
	}
	return len(probabilities) - 1
}

func AddConstant(x []float32, c float32) {
	for i := range x {
		x[i] += c
	}
	return
}

func Sum(x []float32) (y float32) {
	for i := range x {
		y += x[i]
	}
	return
}
func Normalize(x []float32) {
	var sum float32
	for i := range x {
		sum += x[i]
	}
	for i := range x {
		x[i]/=sum
	}
	return
}

func Max(x, y int) int {
	if x >= y {
		return x
	}
	return y
}
func Min(x, y int) int {
	if x >= y {
		return y
	}
	return x
}

func ColumnSums(x [][]float32) (y []float32) {
	y = make([]float32, len(x[0]))
	for i := range y {
		for j := range x {
			y[i] += x[j][i]
		}
	}
	return
}
func EMul(x, y []float32) (z []float32) {
	z = make([]float32, len(x))
	for i := range y {
		z[i] = x[i] * y[i]
	}
	return
}
func ColumnSumsAddConstant(x [][]float32, c float32) (y []float32) {
	y = make([]float32, len(x[0]))
	for i := range y {
		for j := range x {
			y[i] += x[j][i]
		}
		y[i] += c
	}
	return
}
func XAddConstantDivY(x []float32, c float32, y []float32) (z []float32) {
	z = make([]float32, len(x))
	for i := range z {
		z[i] = (x[i] + c) / y[i]
	}
	return
}
func XAddConstantDivConstant(x []float32, c1 float32, c2 float32) (z []float32) {
	z = make([]float32, len(x))
	for i := range z {
		z[i] = (x[i] + c1) / c2
	}
	return
}

func Split(str string) []string {
	str = strings.ToLower(str)
	var split []string
	token := ""
	for _, r := range str {
		switch {
		case unicode.IsPunct(r) || unicode.IsSymbol(r):
			if len(token) > 0 {
				split = append(split, token)
				token = ""
			}
			split = append(split, string(r))
		case len(token) == 0 && unicode.IsSpace(r):
			continue // skip leading space
		case len(token) == 0 && !unicode.IsSpace(r):
			token = string(r)
		case len(token) > 0 && !unicode.IsSpace(r):
			token += string(r)
		case len(token) > 0 && unicode.IsSpace(r):
			split = append(split, token)
			token = ""
		default:
			panic(fmt.Errorf("unknown symbol %q", r))
		}
	}
	if len(token) > 0 {
		split = append(split, token)
	}
	return split
}
