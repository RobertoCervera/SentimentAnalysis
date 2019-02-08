
# coding: utf-8

# # Google W2V:
# ### https://code.google.com/archive/p/word2vec/
#
# # Sandford W2V (GloVe):
# ### https://nlp.stanford.edu/projects/glove/
#
# # FastTex
# ### https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md
#
# ### Example:
#
#    ###  https://blog.manash.me/how-to-use-pre-trained-word-vectors-from-facebooks-fasttext-a71e6d55f27

import numpy as np

import gensim
from keras.layers.embeddings import Embedding


class WordEmbedding():

    wordToIndex = {}
    wordToVecMap = {}
    vecWordSize = 300

    def ReadGloVeVecs(self,gloveSize=200,source="wikipedia"):
        """
        Reads the standfors's word2vec (GloVe)

        Arguments:

        gloveSize -- size of the vector per word
        source -- source from wich words were gathered
        """


        gloveSources = {
        "twitter": "/home/ksquare/Documents/SentimentAnalysis/English/word2vec/glove/twitter/glove.twitter.27B.{}d.txt".format((gloveSize)),
        "wikipedia": "/home/ksquare/Documents/SentimentAnalysis/English/word2vec/glove/wikipedia/glove.6B.{}d.txt".format((gloveSize)),
                }

        gloveFile = gloveSources.get(source)

        gloveFile = "/home/ksquare/Documents/SentimentAnalysis/English/word2vec/SBW-vectors-300-min5.txt"


        self.vecWordSize = gloveSize

        with open(gloveFile, 'r') as f:

            words = set()
            self.wordToVecMap = {}

            for line in f:

                line = line.strip().split()
                currentWord = line[0]
                words.add(currentWord)
                self.wordToVecMap[currentWord] = np.array(line[1:], dtype=np.float64)

            self.wordToIndex = {}

            for index,w in enumerate(sorted(words)):

                self.wordToIndex[w] = index + 1

    def ReadGoogleVec(self):
        """
        Reads the google's word2vec (GloVe)

        """


        en_model = gensim.models.KeyedVectors.load_word2vec_format('/home/ksquare/Documents/SentimentAnalysis/English/word2vec/genism/GoogleNews-vectors-negative300.bin', binary=True)

        words = set()
        self.wordToVecMap = {}

        for word in en_model.vocab:
            words.add(word)
            self.wordToVecMap[word] = np.array(en_model[word], dtype=np.float64)


        self.wordToIndex = {}

        for index,w in enumerate(sorted(words)):

            self.wordToIndex[w] = index + 1


    def ReadFacebookVec(self):

        en_model = gensim.models.KeyedVectors.load_word2vec_format('/home/ksquare/Documents/AI/SentimentAnalysis/English/word2vec/Facebook/Spanish/wiki.es.vec')

        words = set()
        self.wordToVecMap = {}

        for word in en_model.vocab:
            words.add(word)
            self.wordToVecMap[word] = np.array(en_model[word], dtype=np.float64)


        self.wordToIndex = {}

        for index,w in enumerate(sorted(words)):

            self.wordToIndex[w] = index + 1

    def AddUnknowWord(self):

        self.wordToIndex['unknow'] = 0
        self.wordToVecMap['unknow'] = np.zeros(self.vecWordSize)


    def sentencesToIndices(self,sentences,maxLenOfSentence=45):
        """
        Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
        The output shape should be such that it can be given to `Embedding()` (described in Figure 4).

        Arguments:
        sentences -- array of sentences (strings), of shape (m, 1)
        wordToIndex -- a dictionary containing the each word mapped to its index
        maxLenOfSentence -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this.

        Returns:
        indicesOfSentences -- array of indices corresponding to words in the sentences from X, of shape (m, maxLenOfSentence)
        """

        m = sentences.shape[0]
        indicesOfSentences = np.zeros([m,maxLenOfSentence])

        for i in range(m):

            # Convert the ith training sentence in lower case and split is into words. You should get a list of words.
            sentenceWords = (sentences[i].lower()).split()

            # Initialize j to 0
            j = 0

            # Loop over the words of sentence_words
            for word in sentenceWords:
                # Set the (i,j)th entry of X_indices to the index of the correct word.

                try:
                    indicesOfSentences[i, j] = self.wordToIndex[word]
                    j = j+1

                except:
                    kl=0
                #j = j+1

        return indicesOfSentences


    def PretrainedEmbeddingLayer(self):
        """
        Creates a Keras Embedding() layer and loads in pre-trained word2vector.

        Arguments:
        wordToVecMap -- dictionary mapping words to their GloVe vector representation.
        wordToIndex -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

        Returns:
        embeddingLayer -- pretrained layer Keras instance
        """

        vocabLen = len(self.wordToIndex) + 2                  # adding 1 to fit Keras embedding (requirement)
        embDim = self.wordToVecMap["cucumber"].shape[0]      # define dimensionality of your word2vec word vectors
        ### START CODE HERE ###
        # Initialize the embedding matrix as a numpy array of zeros of shape (vocabLen, dimensions of word vectors = embDim)
        embMatrix = np.zeros([vocabLen,embDim])

        print(np.shape(embMatrix))
        # Set each row "index" of the embedding matrix to be the word vector representation of the "index"th word of the vocabulary
        for word, index in self.wordToIndex.items():

            embMatrix[index, :] = self.wordToVecMap[word]

        # Define Keras embedding layer with the correct output/input sizes, make it trainable. Use Embedding(...). Make sure to set trainable=False.
        embeddingLayer = Embedding(vocabLen,embDim,trainable=False)
        ### END CODE HERE ###

        # Build the embedding layer, it is required before setting the weights of the embedding layer. Do not modify the "None".
        embeddingLayer.build((None,))

        # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
        embeddingLayer.set_weights([embMatrix])

        return embeddingLayer
