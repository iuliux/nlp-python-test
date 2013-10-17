import os, fnmatch, nltk, random, sys, re, time
from parser import Parser

VERBOSE = True

def debug(msg):
    if VERBOSE:
        print msg

def findFiles(path, filter):
    for root, dirs, files in os.walk(path):
        for file in fnmatch.filter(files, filter):
            yield os.path.join(root, file)

def findAttr(words, type, attribute):
    for w in words:
        val = w[1][type]
        if val == attribute:
            return True
            
def extractFeatures(text, file):
    fileName = file.split('/')[2]

    features = {#Is the word <guarantee> negated?
                'guaranteeHasNegation':False, 
                #Is there "Low risk" or "No risk" near <guarantee> (a few words away / in the same sentence)?
                'guaranteeLowRisk':False, 
                #If the word <guarantee> appears with the words "we" or "our"
                'guaranteeOur':False,
                #If the word <guarantee> appears with a percentage figure
                'guaranteePercentage':False,
                #If the word <guarantee> appears with the lemma "return"
                'guaranteeReturn':False,
                #If the word <guaranteed> appeared with the words "investment" or "income"
                'guaranteeInvestmentOrIncome':False,
                #If the word <guaranteed> appears with the word capital
                'guaranteeCapital':False,
                
                }
    
    #debug("Processing %s" % file)
    try:
        p = Parser("./json-cache");
        nlpResult = p.parse(fileName, text)
        
        #look for the lemma "guarantee" or "guaranteed"
        for (s) in nlpResult['sentences']:
            #print s['words']
            #print s['parsetree']
            #print s['dependencies']
            #print s['text']
            
            hasGuarantee = False
            #look for trigger words
            for (w) in s['words']:
                lemma = w[1]['Lemma']
                if lemma == 'guarantee' or lemma == 'guaranteed':
                    hasGuarantee = True
                    break
                    
            if hasGuarantee:
                #look for negation
                negations = ['no', 'not', 'isn\'t', 'nothing', 'neither', 'nor']
                if re.search(r'\\b' + '\\b|\\b'.join(negations) + '\\b', s['text'], re.IGNORECASE) != None:
                    features['guaranteeHasNegation'] = True                        
                
                #check for percentage
                if re.search(r'%', s['text']) != None:
                    features['guaranteePercentage'] = True                        
                
                #check for low risk
                if re.search(r'low risk|no risk', s['text'], re.IGNORECASE) != None:
                    features['guaranteeLowRisk'] = True
            
                for (w) in s['words']:
                    lemma = w[1]['Lemma']
                    if lemma == 'our' or lemma == 'we':
                        features['guaranteeOur'] = True
                        
                    if lemma == 'return':
                        features['guaranteeReturn'] = True                        
                        
                    if lemma == 'investment' or lemma == 'income':
                        features['guaranteeInvestmentOrIncome'] = True
                        
                    if lemma == 'capital':
                        features['guaranteeCapital'] = True                        
                        
    except Exception: 
        print " ===== ERROR ====="
        print sys.exc_info()
        return features
    
    return features
            
#read in raw content
folder = './raw-data'
rawData = []
for textFile in findFiles(folder, '*.txt'):
    textData = open(textFile, 'r').read()
    label = "valid" if textFile.find("-n") != -1 else "invalid"
    #print "\n" + textFile + "\n" + label + '\n'
    rawData.append([textData, label, textFile])
        

#Determine training, test and dev sets
size = int(round((len(rawData) * 0.1), 0))
testData = rawData[:size]
devData = rawData[size:]

random.shuffle(devData)
devTestData = devData[:size]
trainData = devData[size:]
        
trainSet = [(extractFeatures(n, f), g) for (n,g,f) in trainData]
devTestSet = [(extractFeatures(n, f), g) for (n,g,f) in devTestData]
testSet = [(extractFeatures(n, f), g) for (n,g,f) in testData]        

#print trainSet
print "Raw items %d / Train items %d / Dev items %d / Test items %d \n" % (len(rawData), len(trainSet), len(devTestSet), len(testSet))


print "----- NaiveBayesClassifier -----\n"
nb_classifier = nltk.NaiveBayesClassifier.train(trainSet)
nb_classifier.show_most_informative_features(10)
print "Accuracy " + str(nltk.classify.accuracy(nb_classifier, devTestSet))

# Test against the dev set, so that we can refine our features
errors = []
debug ("\n ===== TEST RESULTS =====")
for (text, label, file) in devTestData:
    #print file + " - " + label
    guess = nb_classifier.classify(extractFeatures(text, file))
    content = open(file, 'r').read()
        
    if guess != label:
        errors.append( (label, guess, f) )
        debug("correct=%-8s guess=%-8s file=%-30s" % (label, guess, file))
    
debug ("Errors: %d \n" % (len(errors)))



print "----- DecisionTreeClassifier -----\n"
dt_classifier = nltk.DecisionTreeClassifier.train(trainSet)
print dt_classifier.pseudocode()
print "DecisionTreeClassifier Accuracy " + str(nltk.classify.accuracy(dt_classifier, devTestSet))

# Test against the dev set, so that we can refine our features
errors = []
debug ("\n ===== TEST RESULTS =====")
for (text, label, file) in devTestData:
    #print file + " - " + label
    guess = dt_classifier.classify(extractFeatures(text, file))
    content = open(file, 'r').read()
        
    if guess != label:
        errors.append( (label, guess, f) )
        debug("correct=%-8s guess=%-8s file=%-30s" % (label, guess, file))
    
debug ("Errors: %d \n" % (len(errors)))