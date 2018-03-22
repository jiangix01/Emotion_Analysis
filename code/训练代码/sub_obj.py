#coding:utf-8


import sys
default_encoding="utf-8"
if(default_encoding!=sys.getdefaultencoding()):
	reload(sys)
	sys.setdefaultencoding(default_encoding)
	
def text():

	f1 = open('sub.txt','r')

	f2 = open('obj.txt','r')

	line1 = f1.readline()

	line2 = f2.readline()

	str = ''

	while line1:

		str += line1

		line1 = f1.readline()

	while line2:

		str += line2

		line2 = f2.readline()

	f1.close()

	f2.close()

	return str



def bag_of_words(words):

	return dict([(word,True) for word in words])


import nltk

from nltk.collocations import  BigramCollocationFinder

from nltk.metrics import  BigramAssocMeasures



def  bigram(words,score_fn=BigramAssocMeasures.chi_sq,n=1000):

	bigram_finder=BigramCollocationFinder.from_words(words)

	bigrams = bigram_finder.nbest(score_fn,n)

	newBigrams = [u+v for (u,v) in bigrams]

	return bag_of_words(newBigrams)



def  bigram_words(words,score_fn=BigramAssocMeasures.chi_sq,n=1000):

	bigram_finder=BigramCollocationFinder.from_words(words)

	bigrams = bigram_finder.nbest(score_fn,n)

	newBigrams = [u+v for (u,v) in bigrams]

	a = bag_of_words(words)

	b = bag_of_words(newBigrams)

	a.update(b)

	return a

import jieba


def read_file(filename):

	stop = [line.strip() for line in  open('stop.txt','r').readlines()]#停用词

	f = open(filename,'r')

	line = f.readline()

	str = []

	while line:

		s = line.split('\t')

		fenci = jieba.cut(s[0],cut_all=False)

		str.append(list(set(fenci)-set(stop)))
		
		line = f.readline()

	return str



from nltk.probability import  FreqDist,ConditionalFreqDist

from nltk.metrics import  BigramAssocMeasures


def build_features():


	feature =  bigram_words(text(),score_fn=BigramAssocMeasures.chi_sq,n=500)


	posFeatures = []

	for items in read_file('sub.txt'):

		a = {}

		for item in items:

			if item in feature.keys():

				a[item]='True'

		posWords = [a,'pos'] 

		posFeatures.append(posWords)


	negFeatures = []

	for items in read_file('obj.txt'):

		a = {}

		for item in items:

			if item in feature.keys():

				a[item]='True'

		negWords = [a,'neg'] 

		negFeatures.append(negWords)

	return posFeatures,negFeatures


posFeatures,negFeatures =  build_features()


from random import shuffle

shuffle(posFeatures) 

shuffle(negFeatures) 


train =  posFeatures[100:]+negFeatures[100:]

test = posFeatures[:100]+negFeatures[:100]

data,tag = zip(*test)


def score(classifier):

	classifier = SklearnClassifier(classifier) 

	classifier.train(train)


	pred = classifier.classify_many(data) 

	n = 0
	
	m=0

	s = len(pred)

	for i in range(0,s):

		if pred[i]==tag[i] and pred[i]=='neg':

			n = n+1
		
		if pred[i]=='neg':
			
			m=m+1

	return float(n)/float(m) 

import sklearn


from nltk.classify.scikitlearn import  SklearnClassifier

from sklearn.svm import SVC, LinearSVC,  NuSVC

from sklearn.naive_bayes import  MultinomialNB, BernoulliNB

from sklearn.linear_model import  LogisticRegression

from sklearn.metrics import  accuracy_score


print('LogisticRegression`s accuracy is  %f' %score(LogisticRegression()))

print('SVC`s accuracy is %f'  %score(SVC()))

print('LinearSVC`s accuracy is %f'  %score(LinearSVC()))

print('NuSVC`s accuracy is %f'  %score(NuSVC()))