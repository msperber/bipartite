'''
Created on Dec 25, 2013

@author: Matthias Sperber
'''

import numpy as np
import utility
import prob
import math
import expressions as expr

class GibbsParameters(object):
    def __init__(self, numIterations):
        self.numIterations = numIterations

class HyperParameters(object):
    def __init__(self, alphaTheta, alphaF):
        self.alphaTheta = alphaTheta
        self.alphaF = alphaF

class GibbsSamplingVariables(object):
    def __init__(self, textCorpus, nTopics = 1):
        self.textCorpus = textCorpus
        self.allocateVars(textCorpus, nTopics)
        self.initWithFullTopicsAndGammasFromFrequencies(textCorpus, nTopics)
        
    def allocateVars(self, textCorpus, nTopics):
        vocabSize = textCorpus.getVocabSize()
        self.u = np.empty((vocabSize, nTopics)) # scores for word-types in topics
        self.z = np.empty((vocabSize, nTopics), dtype=np.int8) # which word-types belong to which topics
#        self.f = np.empty((vocabSize, nTopics)) # topic weights
        self.t = [] # topic assignments
        for doc in textCorpus:
            self.t.append(np.empty(len(doc)))
#        self.theta = np.empty((len(textCorpus),nTopics)) # topic proportions
        self.gammas = np.empty((vocabSize,)) # reading interest ("word popularity")
        self.w = np.empty((nTopics,)) # topic popularity
        
    def initWithFullTopicsAndGammasFromFrequencies(self, textCorpus, nTopics):
        # initialize variables to a consistent state
        # ensure cosistency by making all words belong to all topics initially
#        vocabSize = textCorpus.getVocabSize()
        self.u.fill(0.5)
        self.z.fill(1)
#        self.f.fill(1.0/vocabSize)
        for docIndex in range(len(textCorpus)):
            self.t[docIndex] = np.random.randint(0, nTopics, len(self.t[docIndex]))
#        self.theta.fill(1.0/nTopics)
        wordFreqs = textCorpus.getWordFrequencies()
        totalNumWords = textCorpus.getTotalNumWords()
        for wordTypeIndex in range(textCorpus.getVocabSize()):
            self.gammas[wordTypeIndex,1] = float(wordFreqs[wordTypeIndex]) / float(totalNumWords) 
        self.w.fill(1.0)
    
def getActiveTopicIndices(textCorpus, t, excludeTopicsOnlyContainingWord=None):
    # TODO: make more efficient, e.g. cache
    if excludeTopicsOnlyContainingWord is None:
        return np.unique(t)
    else:
        activeIndices = set()
        for docIndex in range(len(t)):
            for wordIndex in range(len(t[docIndex])):
                if textCorpus[docIndex][wordIndex]!=excludeTopicsOnlyContainingWord:
                    activeIndices.add(t[docIndex][wordIndex])
        return list(activeIndices)
            
def getUnusedTopicIndex(textCorpus, t):
    # TODO: this should be managed more efficiently with a linked list of unused indices
    activeTopics = getActiveTopicIndices(textCorpus, t)
    unusedTopicIndex = 0
    while unusedTopicIndex in activeTopics:
        unusedTopicIndex += 1
    return unusedTopicIndex

def getNumTopicOccurencesInDoc(topic, samplingDoc, t,
                                    excludeDocWords=[]):
    num = 0
    for wordPos in range(len(t[samplingDoc])):
        if t[samplingDoc, wordPos]==topic:
            if (samplingDoc, wordPos) not in excludeDocWords:
                num += 1
    return num

def getNumTopicAssignmentsToWordType(topic, wordType, t,
                                    excludeDocWords=[]):
    num = 0
    for doc in range(len(t)):
        for wordPos in t[doc]:
            if topic==t[doc, wordPos]:
                if (doc, wordPos) not in excludeDocWords:
                    num += 1
    return num

def getNumWordsInTopic(topic, z):
    return z[topic,:].sum()

def getRthActiveWordTypeInTopic(r, topic, z):
    return np.nonzero(z[topic,:])[0][r]

########################
### MAIN ALGORITHM #####
########################


def inferTopicsCollapsedGibbs(textCorpus, hyperParameters, gibbsParameters):
    
    # initialize variables
    samplingVariables = GibbsSamplingVariables(textCorpus, nTopics = 20)
    
    for iteration in range(gibbsParameters.numIterations):
        print "Gibbs sampling iteration:", iteration
        updateUs(samplingVariables)
        updateZs(textCorpus, samplingVariables, hyperParameters)
        updateWGStars()
        updateGammas()
        
########################
### UPDATES ############
########################

def updateUs(textCorpus, samplingVariables):
    """
    follows [Caron, 2012, Section 5]
    """
    for i in range(textCorpus.getVocabSize()):
        for j in samplingVariables.getActiveTopicIndices():
            if utility.approx_equal(samplingVariables.z[i][j], 0.0):
                samplingVariables.z[i][j] = 1.0
            else:
                samplingVariables.z[i][j] = \
                        prob.sampleRightTruncatedExponential(
                                         samplingVariables.gammas[i]*samplingVariables.w[j],
                                         1.0)

def updateZs(textCorpus, samplingVariables, hyperParameters):
    """
    a Metropolis algorithm to update z's and t's simultaneously 
    """
    for i in range(textCorpus.getVocabSize()):
        for j in samplingVariables.getActiveTopicIndices(excludeTopicsOnlyContainingWord=i):
            
            # switch z_ij between 0 and 1
            zTilde = samplingVariables.z.copy()
            zTilde[i,j] = 1 - zTilde[i,j]
            
            # resample invalidated topics
            tTilde = samplingVariables.t.copy()
            LQij = []
            for docIndex in range(len(textCorpus)):
                for wordIndex in range(len(textCorpus[docIndex])):
                    if textCorpus[docIndex][wordIndex]==i \
                            and tTilde[docIndex, wordIndex]==j:
                        LQij.append((docIndex, wordIndex))
            for r in range(len(LQij)):
                docIndex, wordPos = LQij[r]
                tTilde[docIndex, wordPos] = sampleTGivenZT(
                            samplingDoc=docIndex, 
                            samplingWordPos=wordPos,
                            alphaTheta=hyperParameters.alphaTheta, 
                            alphaF=hyperParameters.alphaF,
                            textCorpus=textCorpus,
                            t=tTilde,
                            z=zTilde,
                            excludeDocWords=LQij[r+1:])
            
            # compute relative probabilities
            ratio = min(1.0,
                        computeRelativeProbabilityForTZ(textCorpus=textCorpus, 
                                                        wordType=i, 
                                                        topicIndex=j, 
                                                        t=tTilde, 
                                                        z=zTilde,
                                                        gammas=samplingVariables.gammas, 
                                                        w=samplingVariables.w, 
                                                        alphaTheta=hyperParameters.alphaTheta, 
                                                        alphaF=hyperParameters.alphaF) \
                        / computeRelativeProbabilityForTZ(textCorpus=textCorpus, 
                                                        wordType=i, 
                                                        topicIndex=j, 
                                                        t=samplingVariables.t, 
                                                        z=samplingVariables.z,
                                                        gammas=samplingVariables.gammas, 
                                                        w=samplingVariables.w, 
                                                        alphaTheta=hyperParameters.alphaTheta, 
                                                        alphaF=hyperParameters.alphaF))
            # accept or reject
            if prob.flipCoin(ratio):
                samplingVariables.z = zTilde
                samplingVariables.t = tTilde

        # create new topics
        numNewTopics = sampleTruncatedNumNewTopics(textCorpus=textCorpus, 
                                          t=samplingVariables.t, 
                                          alphaTheta=hyperParameters.alphaTheta, 
                                          wordType=i,
                                          gammas=samplingVariables.gammas,
                                          alpha=hyperParameters.alpha, 
                                          sigma=hyperParameters.sigma, 
                                          tau=hyperParameters.tau)
        for _ in range(numNewTopics):
            newTopicIndex = getUnusedTopicIndex(textCorpus, samplingVariables.t)
            # TODO:
            # expand z matrix if necessary
            # fill the new row with all zeros, except for word i for which it should be 1
        

def updateWGStars():
    pass

def updateGammas():
    pass

########################
### SAMPLING ###########
########################

def sampleTGivenZT(samplingDoc, samplingWordPos, alphaTheta, alphaF, textCorpus, t, z,
                   excludeDocWords=[]):
    unnormalizedTopicProbs = []
    wordType = textCorpus[samplingDoc][samplingWordPos]
    for topic in getActiveTopicIndices(textCorpus, t):
        if z[topic, wordType]==0:
            unnormalizedTopicProbs.append(0.0)
        else:
            numerator1 = alphaTheta + getNumTopicOccurencesInDoc(topic, samplingDoc, t,
                                    excludeDocWords=[(samplingDoc,topic)] + excludeDocWords)
            numerator2 = np.gamma(alphaF + getNumTopicAssignmentsToWordType(topic, wordType, t,
                                    excludeDocWords=[(samplingDoc,topic)] + excludeDocWords))
            denominator = sum([alphaF + getNumTopicAssignmentsToWordType( \
                                    getRthActiveWordTypeInTopic(r, topic, z), topic, 
                                    excludeDocWords=[(samplingDoc,topic)] + excludeDocWords) \
                               for r in range(getNumWordsInTopic(topic, z))])
            unnormalizedTopicProbs.append(numerator1 * numerator2 / denominator)
    normalizer = sum(unnormalizedTopicProbs)
    normalizedTopicProbs = [p / normalizer for p in unnormalizedTopicProbs]
    return np.nonzero(np.random.multinomial(1, normalizedTopicProbs))[0][0]

def computeRelativeProbabilityForTZ(textCorpus, wordType, topicIndex, t, z, gammas, w, 
                                    alphaTheta, alphaF):
    if oneIfTopicAssignmentsSupported(textCorpus, t, z)!=1:
        return 0
    
    factor1 = (1.0 - math.exp(gammas[wordType]*w[topicIndex]))**z[topicIndex,wordType]
    
    factor2 = math.exp(-(1-z[topicIndex,wordType])*gammas[wordType]*w[topicIndex])
    
    factor3 = 1.0
    activeTopics = getActiveTopicIndices(textCorpus, t)
    for docIndex in range(len(textCorpus)):
        subNumerator1 = np.gamma(len(activeTopics) * alphaTheta)
        subDenominator1 = np.gamma(alphaTheta) ** len(activeTopics)
        subNumerator2 = 1.0
        for topic in activeTopics:
            subNumerator2 *= np.gamma(alphaTheta + getNumTopicOccurencesInDoc(topic, 
                                                                              docIndex, t))
        subDenominator2 = np.gamma(len(activeTopics)*alphaTheta \
                                   + sum([getNumTopicOccurencesInDoc(topic, docIndex, t) \
                                          for t in activeTopics]))
        factor3 *= subNumerator1 / subDenominator1 * subNumerator2 / subDenominator2
    
    factor4 = 1.0
    for topic in activeTopics:
        subNumerator1 = np.gamma(getNumWordsInTopic(topic, z)*alphaF)
        subDenominator1 = np.gamma(alphaF) ** getNumWordsInTopic(topic, z)
        subNumerator2 = 1.0
        for r in range(getNumWordsInTopic(topic, z)):
            subNumerator2 *= np.gamma(alphaF + 
                                      getNumTopicAssignmentsToWordType(topic, 
                                                    getRthActiveWordTypeInTopic(r, topic, z), 
                                                    t))
        subDenominator2 = np.gamma(getNumWordsInTopic(topic, z)*alphaF \
                                   + sum([getRthActiveWordTypeInTopic(r, topic, z) \
                                          for r in range(getNumWordsInTopic(topic, z))]))
        factor4 *= subNumerator1 / subDenominator1 * subNumerator2 / subDenominator2
    return factor1 * factor2 * factor3 * factor4

def oneIfTopicAssignmentsSupported(textCorpus, t, z):
    for docIndex in range(len(t)):
        for wordPos in range(len(t[docIndex])):
            topic = t[docIndex][wordPos]
            wordType = textCorpus[docIndex][wordPos]
            if z[topic, wordType]!=1:
                return 0
    return 1

def sampleTruncatedNumNewTopics(textCorpus, t, alphaTheta, wordType, gammas, alpha, sigma, tau, cutoff=20):
    activeTopicIndices = getActiveTopicIndices(textCorpus, t)
    
    unnormalizedProbs = []
    for num in range(cutoff):
        factor1 = 1.0
        for doc in range(len(textCorpus)):
            numerator = 1.0
            for j in activeTopicIndices:
                numerator *= np.gamma(alphaTheta + \
                                      getNumTopicOccurencesInDoc(topic=j, 
                                                                 samplingDoc=doc, 
                                                                 t=t))
            denominator = np.gamma(len(textCorpus[doc]) + \
                                   (len(activeTopicIndices)+ num) * alphaTheta )
            factor1 *= numerator / denominator
        # TODO: write out Poisson PDF
        k = num
        lam = expr.psiTildeFunction(gammas[wordType], 
                                    sum(gammas)-gammas[wordType],
                                    alpha,
                                    sigma,
                                    tau)
        factor2 = lam**k * math.exp(-lam) / math.factorial(k)
        unnormalizedProbs.append(factor1 * factor2)
    normalizer = sum(unnormalizedProbs)
    normalizedProbs = [p / normalizer for p in unnormalizedProbs]
    return np.nonzero(np.random.multinomial(1, normalizedProbs))[0][0]

        