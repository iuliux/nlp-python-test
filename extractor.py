import os, fnmatch, nltk, random, re
from itertools import chain
from parser import ParseMultiJob

VERBOSE = True
CACHE_DIR = "./json-cache"
N_SERVERS = 2  # Set to 1 for no parallelization

def debug(msg):
    if VERBOSE:
        print msg

def _chunkList(seq, num):
    '''Splits list into @num aprox. equal chunks'''
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out


def findFiles(path, filter):
    for root, dirs, files in os.walk(path):
        for file in fnmatch.filter(files, filter):
            yield os.path.join(root, file)

def findAttr(words, type, attribute):
    for w in words:
        val = w[1][type]
        if val == attribute:
            return True

def parse_datasets(cnkd_trainData, cnkd_devTestData, cnkd_testData):
    # Create parsing jobs
    train_parse_threads = \
            [ParseMultiJob(CACHE_DIR, cnkd_trainData[srv_idx], server=srv_idx)
                for srv_idx in range(N_SERVERS)]
    devTest_parse_threads = \
            [ParseMultiJob(CACHE_DIR, cnkd_devTestData[srv_idx], server=srv_idx)
                for srv_idx in range(N_SERVERS)]
    test_parse_threads = \
            [ParseMultiJob(CACHE_DIR, cnkd_testData[srv_idx], server=srv_idx)
                for srv_idx in range(N_SERVERS)]
    # Start all jobs
    [job.start() for job in train_parse_threads]  # Start jobs
    [job.join() for job in train_parse_threads]  # Wait for jobs to finish
    [job.start() for job in devTest_parse_threads]  # Start jobs
    [job.join() for job in devTest_parse_threads]  # Wait for jobs to finish
    [job.start() for job in test_parse_threads]  # Start jobs
    [job.join() for job in test_parse_threads]  # Wait for jobs to finish
    # Collect results
    train_results = list(chain.from_iterable([job.nlpResults for job in train_parse_threads]))
    devTest_results = list(chain.from_iterable([job.nlpResults for job in devTest_parse_threads]))
    test_results = list(chain.from_iterable([job.nlpResults for job in test_parse_threads]))

    return train_results, devTest_results, test_results

def extractFeatures(parsed):
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
    
    #look for the lemma "guarantee" or "guaranteed"
    for (s) in parsed['sentences']:
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

# Chunk data to be processed on multiple servers
cnkd_trainData = _chunkList(trainData, N_SERVERS)
cnkd_devTestData = _chunkList(devTestData, N_SERVERS)
cnkd_testData = _chunkList(testData, N_SERVERS)

# Parse datasets
train_results, devTest_results, test_results = \
    parse_datasets(cnkd_trainData, cnkd_devTestData, cnkd_testData)

# Extract features
trainSet = [(extractFeatures(parsed), g) for (parsed, g, f) in train_results]
devTestSet = [(extractFeatures(parsed), g) for (parsed, g, f) in devTest_results]
testSet = [(extractFeatures(parsed), g) for (parsed, g, f) in test_results]

# # print trainSet
print "Raw items %d / Train items %d / Dev items %d / Test items %d \n" % (len(rawData), len(trainSet), len(devTestSet), len(testSet))


print "----- NaiveBayesClassifier -----\n"
nb_classifier = nltk.NaiveBayesClassifier.train(trainSet)
nb_classifier.show_most_informative_features(10)
print "Accuracy " + str(nltk.classify.accuracy(nb_classifier, devTestSet))

# Test against the dev set, so that we can refine our features
errors = []
debug ("\n ===== TEST RESULTS =====")
for (text, label, fname) in devTest_results:
    #print file + " - " + label
    guess = nb_classifier.classify(extractFeatures(text))
        
    if guess != label:
        errors.append( (label, guess) )
        debug("correct=%-8s guess=%-8s file=%-30s" % (label, guess, fname))
    
debug ("Errors: %d \n" % (len(errors)))



print "----- DecisionTreeClassifier -----\n"
dt_classifier = nltk.DecisionTreeClassifier.train(trainSet)
print dt_classifier.pseudocode()
print "DecisionTreeClassifier Accuracy " + str(nltk.classify.accuracy(dt_classifier, devTestSet))

# Test against the dev set, so that we can refine our features
errors = []
debug ("\n ===== TEST RESULTS =====")
for (text, label, fname) in devTest_results:
    #print file + " - " + label
    guess = dt_classifier.classify(extractFeatures(text))
        
    if guess != label:
        errors.append( (label, guess) )
        debug("correct=%-8s guess=%-8s file=%-30s" % (label, guess, fname))
    
debug ("Errors: %d \n" % (len(errors)))