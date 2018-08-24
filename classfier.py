import nltk
from nltk.stem.lancaster import LancasterStemmer
import pandas as pd
import numpy as np
class classifier():
    training_data = []
    corpus_words = {}
    class_words = {}
    stemmer = LancasterStemmer()

    def training(self):
        dataset = pd.read_csv("dataset.csv")
        for intent in dataset['intent'].unique():
                for text in dataset[dataset['intent']==intent]["Data"]:
                    self.training_data.append({"class":intent,"sentence":text})    
        # word stemmer
        # turn a list into a set (of unique items) and then a list again (this removes duplicates)
        classes = list(set([a['class'] for a in self.training_data]))
        for c in classes:
        # prepare a list of words within each class
            self.class_words[c] = []

        # loop through each sentence in our training data
        for data in self.training_data:
        # tokenize each sentence into words
            for word in nltk.word_tokenize(data['sentence']):
            # ignore a some things
                if word not in ["؟"]:
                    # stem and lowercase each word
                    stemmed_word = self.stemmer.stem(word.lower())
                    # have we not seen this word already?
                    if stemmed_word not in self.corpus_words:
                        self.corpus_words[stemmed_word] = 1
                    else:
                        self.corpus_words[stemmed_word] += 1

                    # add the word to our words in class list
                    self.class_words[data['class']].extend([stemmed_word])

    # calculate a score for a given class taking into account word commonality
    def calculate_class_score(self,sentence, class_name, show_details=True):
        score = 0
        # tokenize each word in our new sentence
        for word in nltk.word_tokenize(sentence):
            # check to see if the stem of the word is in any of our classes
            if self.stemmer.stem(word.lower()) in self.class_words[class_name]:
                # treat each word with relative weight
                score += (1 /self.corpus_words[self.stemmer.stem(word.lower())])*(1/self.class_words[class_name].count(self.stemmer.stem(word.lower())))

                if show_details:
                    print (" match: %s (%s) * (%s)" % (self.stemmer.stem(word.lower()),
                                                    1 / self.corpus_words[self.stemmer.stem(word.lower())],
                                                    (1/self.class_words[class_name].count(self.stemmer.stem(word.lower())))))
        return score

    def trace(self,sentence):
        # now we can find the class with the highest score
        for c in self.class_words.keys():
            print ("Class: %s  Score: %s \n" % (c, self.calculate_class_score(sentence, c)))

    def classify(self,sentence):
        high_class = None
        high_score = 0
        # loop through our classes
        for c in self.class_words.keys():
            # calculate score of sentence for each class
            score = self.calculate_class_score(sentence, c, show_details=False)
            # keep track of highest score
            if score > high_score:
                high_class = c
                high_score = score

        return high_class, high_score


