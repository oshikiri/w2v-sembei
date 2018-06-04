w2v-sembei : Segmentation-free version of word2vec
=======================================================

**w2v-sembei** [1] is a C++ implementation of word segmentation-free version of word2vec [2].


## How to use

It requires gcc(>=5).

```sh
git clone https://github.com/oshikiri/w2v-sembei.git --recursive
cd w2v-sembei/
mkdir output
make
./w2v-sembei 1000 10000 10000 10000 --corpus sample.txt --window 1 --dim 50
```

The outputs are

- list of n-grams (`output/vocabulary.csv`)
- vector representation of n-grams (`output/embeddings_words.csv`)


## References

1. Oshikiri, T. (2017). **Segmentation-Free Word Embedding for Unsegmented Languages**. In Proceedings of EMNLP2017. [[pdf](http://aclweb.org/anthology/D17-1080), [bib](http://aclweb.org/anthology/D17-1080.bib)]
2. Mikolov, T., Corrado, G., Chen, K., & Dean, J. (2013). **Efficient Estimation of Word Representations in Vector Space**. In Proceedings of ICLR2013. [[code](https://code.google.com/archive/p/word2vec/)]
3. [shimo-lab/sembei - GitHub](https://github.com/shimo-lab/sembei)
