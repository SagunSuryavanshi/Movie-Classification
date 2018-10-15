import nltk							
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB, BernoulliNB                  #variation of naive bayes classifier
from sklearn.linear_model import LogisticRegression, SGDClassifier              
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode                                                 #method to find most popular vote

    """
category=pos or neg;
fileid=each review has its own id
In each category take the fileid then store the tokenized word version
for the fileid followed by positive or negative lable in one big list
"""
    
    
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)
#the data is kept in the order of neg pos
#we need to shuffle it inorder to ensure we get a mixed bunch
#print(documents[1])

all_words = []

for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)                                        #to find out the most common words

"""
print(all_words.most_common(15))
#to fnd out the occurences of the word stupid
print(all_words["stupid"])
"""

word_features = list(all_words.keys())[:3000]                               #contains the top 3000 most common words

#Converting Words to Features
#next we find these 3000 words in our positive and negative documents

def find_features(document):
    words = set(document)                                                   #converting from list to set(only words retrived not amount)
    features = {}                                                           #empty dictionary
    for w in word_features:
        features[w] = (w in words)                                          #creates boolean of either True or False so if one of those top 3000 words is in this document then its True

    return features

#print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))         

#saving the features as a dictionary by determining the presence of the top 3000 words and whether or not they are 
#contained  in each review ie running above on all reviews = feature set

featuresets = [(find_features(rev), category) for (rev, category) in documents]
        
#splitting the data into training and testing

training_set = featuresets[:1900]
testing_set =  featuresets[1900:]

"""Naive Bayes 
a family of algorithms that all share a common principle, that every feature being classified is independent of the value of any other feature. So for example, a fruit may be considered to
be an apple if it is red, round, and about 3″ in diameter. A Naive Bayes classifier considers each of these “features” (red, round, 3” in diameter) to contribute independently to the probability
that the fruit is an apple, regardless of any correlations between features. Features, however, aren’t always independent which is often seen as a shortcoming of the Naive Bayes algorithm and
this is why it’s labeled “naive”.
"""


#classifier = nltk.NaiveBayesClassifier.train(training_set)             #to train- first invoke the Naive Bayes Classifier then train it by .train()

#Saving Classifiers with NLTK using Pickle
#We use pickle in order to avoid firing up and training the same model again and again

save_classifier = open("naivebayes.pickle", "wb")        
pickle.dump(classifier,save_classifier)                            
save_classifier.close()        

#to open and use this classifier
classifier_f = open("naivebayes.pickle","rb")                           #reading from the file
classifier = pickle.load(classifier_f)                                 
classifier_f.close()



print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)                           #to find most informative features: (also tells whether pos or neg)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

##SVC_classifier = SklearnClassifier(SVC())
##SVC_classifier.train(training_set)
##print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)

#Combing Algorithms and creating a voting system

class VoteClassifier(ClassifierI):                                     #inheriting from nltk ClassifierI
    def __init__(self, *classifiers):                                  #assign the list of classifiers that are passed to our class self._classifiers
        self._classifiers = classifiers

    def classify(self, features):                                      #create a classify method so that we can call .classify on our classifier
        votes = []
        for c in self._classifiers:                                    #we iterate through all classifiers
            v = c.classify(features)                                   #for each classifier we get vote
            votes.append(v)
        return mode(votes)                                             #who got the most votes

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))                         #how many most popular occurences were in that list
        conf = choice_votes / len(votes)
        return conf

voted_classifier = VoteClassifier(classifier,
                                  NuSVC_classifier,
                                  LinearSVC_classifier,
                                  SGDClassifier_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier)

print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)

print("Classification:", voted_classifier.classify(testing_set[0][0]), "Confidence %:",voted_classifier.confidence(testing_set[0][0])*100)
print("Classification:", voted_classifier.classify(testing_set[1][0]), "Confidence %:",voted_classifier.confidence(testing_set[1][0])*100)
print("Classification:", voted_classifier.classify(testing_set[2][0]), "Confidence %:",voted_classifier.confidence(testing_set[2][0])*100)
print("Classification:", voted_classifier.classify(testing_set[3][0]), "Confidence %:",voted_classifier.confidence(testing_set[3][0])*100)
print("Classification:", voted_classifier.classify(testing_set[4][0]), "Confidence %:",voted_classifier.confidence(testing_set[4][0])*100)
print("Classification:", voted_classifier.classify(testing_set[5][0]), "Confidence %:",voted_classifier.confidence(testing_set[5][0])*100)

