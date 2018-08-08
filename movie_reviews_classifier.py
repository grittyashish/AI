import nltk
import random
from nltk.corpus import movie_reviews
import pickle

#documents is a list of tuples used to train and test(by shuffling)
documents = [(list(movie_reviews.words(fileid)),category)
              for category in movie_reviews.categories()
              for fileid in movie_reviews.fileids(category)]
#documents is a list containing tuples(which in turn contains a list(segmented words or a movie review) and a category(pos or neg))
random.shuffle(documents)#documents contains approx 13,000,000 words
print(type(documents))
#print("Printing documents")
print(documents[4])
#print(documents[1])
all_words = []#A massive list of lowercase words
for w in movie_reviews.words():
    all_words.append(w.lower())
#print(all_words)
all_words = nltk.FreqDist(all_words)
#print(all_words.most_common(15))
#print(all_words["stupid"])
#word_features contains the top 3000 words and our taarget thus far is to check whether these top words appear in some document.
#This document is passed as argument into function find_features

word_features = list(all_words.keys())[:3000]
def find_features(document):
    words = set(document)#Remove repeatition
    features = {}
    for w in word_features:#Iterating over the 3000 words
        features[w] = (w in words)#storing the boolean(whether present in document(words in line 24) or not and mapping it into dictionary)
    return features

#print ((find_features(movie_reviews.words('neg/cv000_29416.txt'))))#cv000_29416.txt is inside neg folder and contains review by a single user of a certain movie. This review is word segmented and passed to find_features function
featuresets = [(find_features(rev),category) for (rev, category) in documents]#Since documents is a list containing tuples(review,category). The dictionary returned by find_features are appended into featuresets
#featuresets : [({word_in_review : False or True - whether this word is present in word_features}, neg/pos)]
training_set = featuresets[:1900]
testing_set = featuresets[1900:]

#posterior = prior occurences * likelihood / current evidence
classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Naive Bayes Algo accuracy percent :",(nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

#To save our trained algorithm so that next time we needn't train it later
#We use pickle to save python objects so that we can load them
save_classifier = open("naivebayes.pickle","wb")#wb is write in bytes as python differentiates between string and bytes
pickle.dump(classifier,save_classifier)
save_classifier.close()

# To use it : Now we don't need to train
#classifier_f = open("naivebayes.pickle","rb")
#classifier = pickle.load(classifier_f)
#classifier_f.close()
#pickle can also be used to save the processed data of thi file
