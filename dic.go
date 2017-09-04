package golda

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"sort"
)

type Token struct {
	ID        int
	Token     string
	Frequency int
}

type Dictionary struct {
	Tokens   map[string]*Token
	id2Token []*Token `json: omit`
}

func (d *Dictionary) Top(n int) {
	if d.Len() <= n {
		return
	}

	dic := &Dictionary{Tokens: make(map[string]*Token)}
	d.id2Token = make([]*Token, 0)
	sorted := make([]*Token, len(d.Tokens))
	i := 0
	for _, t := range d.Tokens {
		sorted[i] = t
		i++
	}
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Frequency > sorted[j].Frequency
	})

	for id, t := range sorted {
		t.ID = id
	}

	d.id2Token = make([]*Token, 0)
	dic.Tokens = make(map[string]*Token)
	for _, t := range sorted[:n] {
		dic.Tokens[t.Token] = t
	}

}

func SaveDictionary(name string, dic *Dictionary) error {
	// save MODEL_NAME
	f, err := os.Create(name)
	if err != nil {
		return err
	}
	encoder := json.NewEncoder(f)
	err = encoder.Encode(dic)
	if err != nil {
		return err
	}
	f.Close()
	return nil
}

func LoadDictionary(name string) (*Dictionary, error) {
	if len(name) == 0 {
		return nil, fmt.Errorf("No dictionary file provided! [%s]", name)
	}
	f, e := os.Open(name)
	if e != nil {
		return nil, e
	}
	var m *Dictionary
	decoder := json.NewDecoder(f)
	e = decoder.Decode(&m)
	if e != nil {
		return nil, e
	}
	f.Close()
	return m, nil
}
func (d *Dictionary) String() string {
	str := ""
	for k, v := range d.Tokens {
		str += fmt.Sprintf("%s:%d\n", k, v)
	}
	return str
}

func (d *Dictionary) TokenByID(id int) *Token {
	if len(d.id2Token) == 0 {
		d.id2Token = make([]*Token, len(d.Tokens))
		for _, v := range d.Tokens {
			d.id2Token[v.ID] = v
		}
	}
	return d.id2Token[id]
}

func (d *Dictionary) IDByToken(token string) int {
	return d.Tokens[token].ID
}

//Len number of tokens in dictionary
func (d *Dictionary) Len() int {
	return len(d.Tokens)
}

func DictionaryFromFile(file string) (*Dictionary, error) {
	f, e := os.Open(file)
	if e != nil {
		return nil, e
	}
	r := bufio.NewReader(f)
	dic := &Dictionary{Tokens: make(map[string]*Token)}
	for {
		line, e := r.ReadString('\n')
		if e != nil {
			break
		}

		if len(line) > 0 {
			for _, token := range Split(line) {
				t, ok := dic.Tokens[token]
				if !ok {
					dic.Tokens[token] = &Token{Token: token, Frequency: 1}
				} else {
					t.Frequency++
				}
			}
		}
	}
	f.Close()
	// enumerate
	dic.Enumerate()
	return dic, nil
}

func (d *Dictionary) Enumerate() {
	d.id2Token = make([]*Token, 0)
	sorted := make([]*Token, len(d.Tokens))
	i := 0
	for _, t := range d.Tokens {
		sorted[i] = t
		i++
	}
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Frequency > sorted[j].Frequency
	})

	for id, t := range sorted {
		t.ID = id
	}
}

func (d *Dictionary) String2Digits(text string) []int {
	tokens := Split(text)
	digits := make([]int, len(tokens))
	for i, token := range tokens {
		digits[i] = d.IDByToken(token)
	}
	return digits
}
func (d *Dictionary) Tokens2Digits(tokens []string) []int {
	digits := make([]int, len(tokens))
	for i, token := range tokens {
		digits[i] = d.IDByToken(token)
	}
	return digits
}
