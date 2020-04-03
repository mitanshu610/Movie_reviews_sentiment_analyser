import nltk
from nltk.corpus import movie_reviews
from nltk import FreqDist
from nltk.classify import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
import pickle
import random


dataset = [(list(movie_reviews.words(fileid)), category)
           for category in movie_reviews.categories()
           for fileid in movie_reviews.fileids(category)]

random.shuffle(dataset)

# print(dataset[1])

all_words = [w.lower() for w in movie_reviews.words()]
# print(all_words)

all_words = FreqDist(all_words)
word_features = (list(all_words.keys()))[:3000]


# print(word_features)

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


featureset = [(find_features(rev), category) for (rev, category) in dataset]
training_set = featureset[:1900]
testing_set = featureset[1900:]



classifier_f = open("naive_bayes.pickle","rb")
classifier = pickle.load(classifier_f)
classifier_f.close()
print("Inbuilt Naive Bayes accuracy = ",(nltk.classify.accuracy(classifier, testing_set))*100)
# classifier.show_most_informative_features()

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MN Naive Bayes accuracy = ",(nltk.classify.accuracy(MNB_classifier, testing_set))*100)


BE_classifier = SklearnClassifier(BernoulliNB())
BE_classifier.train(training_set)
print("BE Naive Bayes accuracy = ",(nltk.classify.accuracy(BE_classifier, testing_set))*100)

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression accuracy = ",(nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier accuracy = ",(nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("SVC accuracy = ",(nltk.classify.accuracy(SVC_classifier, testing_set))*100)
