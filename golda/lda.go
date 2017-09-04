package golda

import (
	"math/rand"
	"time"
	"log"
	"sort"
	"math"
	"go/doc"
)

//From previous research, we have found α =50/T and β = 0.01 to work well with many different text collections.

type LDA struct {
	// number of topics
	T int
	// number of words
	W int
	// number of documents
	D int
	//word-topic count matrix
	WT Matrix
	//document-topic count matrix,
	// where the counts correspond to the number of tokens
	// assigned to each topic for each document
	DT         Matrix
	α, β       float32
	corpus     Corpus
	dictionary *Dictionary
}

func (lda *LDA) Init(topic_count int, corpus Corpus, dictionary *Dictionary) *LDA {
	rand.Seed(time.Now().UnixNano())
	lda.T = topic_count
	lda.W = dictionary.Len()
	lda.D = corpus.Len()
	lda.WT = Matrix{}.Resize(lda.W, lda.T)
	lda.DT = Matrix{}.Resize(lda.D, lda.T)
	lda.α = 50.0 / float32(lda.T)
	lda.β = 0.01
	lda.corpus = corpus
	lda.dictionary = dictionary
	// initialize word-topic assignment for every document
	var topic int
	for _, d := range corpus {
		d.ta = make([]int, len(d.tokens))
		for i, word := range d.tokens {
			topic = rand.Intn(lda.T)
			d.ta[i] = topic
			lda.WT.W[word][topic] ++
		}
	}
	// count document-topic assignment for every document
	for i, d := range corpus {
		for _, topic := range d.ta {
			lda.DT.W[i][topic]++
		}
	}
	return lda
}
func (lda *LDA) PrintTopics(n int) {
	denominators := ColumnSums(lda.WT.W)
	for t := range Range(lda.T) {
		var tokens []int
		for wi := range Range(lda.W) {
			if lda.WT.W[wi][t] > 0 {
				tokens = append(tokens, wi)
			}
		}
		sort.Slice(tokens, func(i, j int) bool {
			return lda.WT.W[tokens[i]][t] > lda.WT.W[tokens[j]][t]
		})
		log.Printf("Topic %d: of %d words\n", t, len(tokens))
		for i := range tokens[:Min(n, len(tokens))] {
			log.Printf("\tToken %d %s %f\n", tokens[i], lda.dictionary.TokenByID(tokens[i]).Token, lda.WT.W[tokens[i]][t]/denominators[t])
		}
	}
}
func (lda *LDA) Step() {
	for di, d := range lda.corpus {
		for i, wi := range d.tokens {
			t0 := d.ta[i]
			//	dt[d,t0] <- dt[d,t0]-1 # we don't want to include token w
			// in our document-topic count matrix when sampling for token w
			lda.DT.W[di][t0]--
			//	wt[t0,wid] <- wt[t0,wid]-1 # we don't want to include token w
			// in our word-topic count matrix when sampling for token w
			lda.WT.W[wi][t0]--
			// UPDATE TOPIC ASSIGNMENT FOR EACH WORD -- COLLAPSED GIBBS SAMPLING MAGIC.  Where the magic happens.
			denom_a := Sum(lda.DT.W[di]) + float32(lda.T)*lda.α // number of tokens in document + number topics * alpha
			denom_b := ColumnSumsAddConstant(lda.WT.W, float32(lda.W)*lda.β)

			//p_z <- (wt[,wid] + eta) / denom_b * (dt[d,] + alpha) / denom_a # calculating probability word belongs to each topic
			p_z := EMul(XAddConstantDivY(lda.WT.W[wi], lda.β, denom_b), XAddConstantDivConstant(lda.DT.W[di], lda.α, denom_a))
			Normalize(p_z)
			//t1 <- sample(1:K, 1, prob=p_z/sum(p_z)) # draw topic for word n from multinomial using probabilities calculated above
			sampled_topic := Multinomial(p_z)

			d.ta[i] = sampled_topic // update topic assignment list with newly sampled topic for token w.
			//dt[d, t1] <-dt[d, t1]+1      re-increment document-topic matrix with new topic assignment for token w.
			lda.DT.W[di][sampled_topic]++
			//wt[t1, wid] <-wt[t1, wid]+1 re-increment word-topic matrix with new topic assignment for token w.
			lda.WT.W[wi][sampled_topic]++
			//if t0 != sampled_topic {
			//	log.Printf("doc: %d token: %d moves from topic: %d -> topic: %d", di, wi, t0, sampled_topic)
			//}
		}
	}
}

func (lda *LDA) CorpusLogLikelihood(corpus *Corpus) float32 {
	var total_log_likelihood  float32
	for di, d := range lda.corpus {
		total_log_likelihood += lda.DocumentLogLikelihood(di, d)
	}
	return total_log_likelihood
}

func (lda *LDA) DocumentLogLikelihood(di, doc *Document) float32 {
	num_topics := lda.T
	doc_length := len(doc.tokens)

	// Compute P(z|d) for the given document and all topics.
	prob_topic_given_document := make([]float32,lda.T)
	smoothed_doc_length := float32(doc_length) + lda.α * float32(lda.T)
	// accumulate document topic counts
	for _, t := range doc.ta {
		prob_topic_given_document[t]++
	}
	// document topic probabilities
	for t := range prob_topic_given_document {
		prob_topic_given_document[t] = (prob_topic_given_document[t] + float32(lda.T)) / smoothed_doc_length
	}

	// Get global topic occurrences, which will be used to compute P(w|z).
	global_topic_histogram := sampler.model.GetGlobalTopicHistogram()
	prob_word_given_topic := make([]float32,lda.T)
	log_likelihood := 0.0;

	// A document's log-likelihood is the sum of log-likelihoods
	// of its words.  Compute the likelihood for every word and
	// sum the logs.
	for i, wi := range doc.ta {
		// Get topic_count_distribution of the current word,
		// which will be used to Compute P(w|z).
		word_topic_histogram := sampler.model.GetWordTopicHistogram(iter.Word())

		// Compute P(w|z).
		for t := 0; t < num_topics; t++ {
			prob_word_given_topic[t] =
				(word_topic_histogram[t] + lda.β) /
					(global_topic_histogram[t] + float32(len(doc.tokens)) * lda.β)
		}

		// Compute P(w) = sum_z P(w|z)P(z|d)
		prob_word := 0.0
		for t := 0; t < num_topics; t++ {
			prob_word += prob_word_given_topic[t] * prob_topic_given_document[t]
		}
		log_likelihood += math.Log(prob_word);
	}

	return log_likelihood
}