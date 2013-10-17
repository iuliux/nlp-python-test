'''
Implements Bag-of-Words model with NaiveBayes classifier.
'''


import os, fnmatch, nltk, random
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.probability import FreqDist
from nltk.classify import SklearnClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

VERBOSE = True

def debug(msg):
    if VERBOSE:
        print msg


def findFiles(path, filter):
    for root, dirs, files in os.walk(path):
        for file in fnmatch.filter(files, filter):
            yield os.path.join(root, file)


lmtzr = WordNetLemmatizer()

def tokenize(text):
    # return [lmtzr.lemmatize(tok.strip().replace('.', ''))  # Uncomment this line and comment the following to enable lemmatization
    return [tok.strip().replace('.', '')
                                for tok in nltk.word_tokenize(text)
                                    if tok != ',' and
                                       tok not in stopwords.words('english')]


# Read in raw content
folder = './raw-data'
rawData = []
for textFile in findFiles(folder, '*.txt'):
    textData = open(textFile, 'r').read()
    label = "valid" if textFile.find("-n") != -1 else "invalid"
    #print "\n" + textFile + "\n" + label + '\n'
    rawData.append((textData, label))

print 'Samples counts: -----------------------------'
print '  invalid:', len([d for d in rawData if d[1] == 'invalid'])
print '  valid:', len([d for d in rawData if d[1] == 'valid'])
print '---------------------------------------------'


#Determine training, test and dev sets
size = int(round((len(rawData) * 0.15), 0))
random.shuffle(rawData)
testData = rawData[:size]
trainData = rawData[size:]
random.shuffle(trainData)

# Generate TermFrequency for each doc
trainTF = [(FreqDist(tokenize(text)), tag) for text, tag in trainData]
testTF = [(FreqDist(tokenize(text)), tag) for text, tag in testData]

# Create classifier
pipeline = Pipeline([('tfidf', TfidfTransformer()),
                     ('chi2', SelectKBest(chi2, k=1000)),
                     ('nb', MultinomialNB())])
classif = SklearnClassifier(pipeline)
# Train classifier
classif.train(trainTF)


# Evaluate
testTags = [tag for tf, tag in testTF]
testResults = classif.batch_classify([tf for tf, tag in testTF])

right = 0
for i, tg in enumerate(testTags):
    if testResults[i] == tg:
        right += 1

print 'Results: ------------------------------------'
print testResults
print 'Accuracy:', right / float(len(testTags))
print '---------------------------------------------'
