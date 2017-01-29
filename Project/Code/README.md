# Machine Learning: From Theory to Practise - Course Project

# Amazon book reviews Data Set (https://archive.ics.uci.edu/ml/datasets/Amazon+book+reviews)

## Data
The data consists of 37 139 reviews from readers for the book "The Girl on the Train", Paula Hawkins.
Each text review is accompanied with a rating, from 0 to 5 depending on how much the reader liked the book.

## Challenge
Given a new review, can you predict the rating the reader will make?

## Preprocessing
Several usual nlp cleanup

## Methods

### Bag of Words
We first use a simple bag of words representation of the reviews. In the bag of words model, we do not consider the order of words in texts, and consider that a text is a set of words, each of them being characterized by a measure of its relative frequency (to be defined, e.g. tf-idf)

Then, usual classification algorithms are performed.
The results are poor.

### Word2Vec
In order to improve over the bag of words model, we wish to take into account the word order. We use the word2vec model from Mikolov's [paper](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf), and implement it thanks to the gensim library

We show some properties of what word2vec can achieve in terms of word embedding and its ability to maintain semantic and syntactic relations between words

Having embedded words into vectors, we wish to embed the reviews into vectors. We try 2 methods: 

- Vector averaging: the review vector is the average of its word vectors
- Clustering: Bag of centroids

We find that this approach does not perform better than the bag of word representation.
This is caused by the small size of our dataset. Modern systems using deep learning word embeddings are trained on huge datasets with billions of words, whereas our dataset is quite small in comparision.
Furthermore, while averaging on the wpord vectors, we lose the word order, and then we get closer to the bag of words.

## How to go further?

- A first possibility is to train the word2vec on the reviews from all books in the dataset (here, we used only one).
- Another possibility is to use pretrained word vectors, that is vectors trained on huge datasets by google, and available for free.
