'''
Created on Dec 25, 2013

@author: Matthias Sperber
'''

import numpy as np
import source.utility as utility
import source.prob as prob
import math
import source.expressions as expr
import random
import copy
from setuptools.command.easy_install import samefile

class HyperParameters(object):
    def __init__(self, alpha, sigma, tau, alphaTheta, alphaF):
        self.alphaTheta = alphaTheta
        self.alphaF = alphaF
        assert 0 <= sigma < 1.0
        self.alpha = alpha
        self.sigma = sigma
        self.tau = tau

class GibbsSamplingVariables(object):
    def __init__(self, textCorpus, nTopics = 1):
        self.deadTopics, self.activeTopics = [], []
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
        self.gStar = None
        self.activeTopics = range(nTopics)
        
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
        wordFreqs = textCorpus.getVocabFrequencies()
        totalNumWords = textCorpus.getTotalNumWords()
        for wordTypeIndex in range(textCorpus.getVocabSize()):
            self.gammas[wordTypeIndex] = float(wordFreqs[wordTypeIndex]) / float(totalNumWords) 
        self.w.fill(1.0)
    
    # approach to managing active & dead topics: both are stored in (complementary) lists,
    # which are only changed upon a call of releaseDeadTopics() or createNewTopics()
    # thus, topics with no associated words remain "active" until releaseDeadTopics() gets called
    def getActiveTopics(self):
        return self.activeTopics
    
    def releaseDeadTopics(self):
        nonEmptyIndices = set()
        for docIndex in range(len(self.t)):
            for wordIndex in range(len(self.t[docIndex])):
                nonEmptyIndices.add(self.t[docIndex][wordIndex])
        emptyActiveIndices = set(self.activeTopics).difference(nonEmptyIndices)
        
        self.activeTopics = list(nonEmptyIndices)
        self.deadTopics.extend(emptyActiveIndices)
    
    def createNewTopics(self, numNewTopics):
        # to be on the safe side: init new z's with 1, new u's with 0.5, new w with 1
        # TODO: make more efficient by grouping memory allocations
        newTopicIndices = []
        for _ in range(numNewTopics):
            if len(self.deadTopics)==0:
                newTopicIndex = len(self.activeTopics)
                # expand z
                newZ = np.ones((self.z.shape[0], self.z.shape[1]+1))
                newZ[:,:-1] = self.z
                self.z = newZ
                # expand u
                newU = np.empty((self.u.shape[0], self.u.shape[1]+1))
                for j in range(self.u.shape[1]):
                    newU[j,-1] = 0.5
                newU[:,:-1] = self.u
                self.u = newU
                # expand w
                newW = np.ones((self.w.shape[0]+1))
                newW[:-1] = self.w
                self.w = newW
            else:
                newTopicIndex = self.deadTopics.pop()
                for i in range(self.z.shape[1]):
                    self.z[i,newTopicIndex] = 1
                    self.u[i,newTopicIndex] = 0.5
                self.w[newTopicIndex] = 1.0
            newTopicIndices.append(newTopicIndex)
            self.activeTopics.append(newTopicIndex)
            
        assert self.z.shape == self.u.shape
        return newTopicIndices

def getNumTopicOccurencesInDoc(topic, samplingDoc, t,
                                    excludeDocWords=[]):
    num = 0
    for wordPos in range(len(t[samplingDoc])):
        if t[samplingDoc][wordPos]==topic:
            if (samplingDoc, wordPos) not in excludeDocWords:
                num += 1
    return num

def getNumTopicAssignmentsToWordType(topic, wordType, t, textCorpus, 
                                    excludeDocWords=[]):
    num = 0
    for doc in range(len(t)):
        for wordPos in range(len(t[doc])):
            if topic==t[doc][wordPos] and textCorpus[doc][wordPos]==wordType:
                if (doc, wordPos) not in excludeDocWords:
                    num += 1
    return num

def getNumWordTypesActivatedInTopic(topic, z):
    return z[topic,:].sum()

def getRthActiveWordTypeInTopic(r, topic, z):
    return np.nonzero(z[:,topic])[0][r]

########################
### MAIN ALGORITHM #####
########################


def inferTopicsCollapsedGibbs(textCorpus, hyperParameters, numIterations):
    
    # initialize variables
    samplingVariables = GibbsSamplingVariables(textCorpus, nTopics = 20)
    
    for iteration in range(numIterations):
        print "Gibbs sampling iteration:", iteration
        updateUs(textCorpus=textCorpus, samplingVariables=samplingVariables)
        updateZs(textCorpus, samplingVariables, hyperParameters)
        updateWGStar(textCorpus, samplingVariables, hyperParameters)
        updateGammas(textCorpus, samplingVariables, hyperParameters)
        # TODO: release dead topics
        
########################
### UPDATES ############
########################

def updateUs(textCorpus, samplingVariables):
    """
    follows [Caron, 2012, Section 5]
    """
    for i in range(textCorpus.getVocabSize()):
        for j in samplingVariables.getActiveTopics():
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
        for j in samplingVariables.getActiveTopics():
            # skip the case where only topic j is active for word i: we need at
            # least one topic in which each word is activated
            if samplingVariables.z[i,:].sum()==1 and samplingVariables.z[i,j]==1:
                continue
            
            # switch z_ij between 0 and 1
            zTilde = samplingVariables.z.copy()
            zTilde[i,j] = 1 - zTilde[i,j]
            
            # resample invalidated topics
            tTilde = copy.deepcopy(samplingVariables.t)
            LQij = []
            for docIndex in range(len(textCorpus)):
                for wordIndex in range(len(textCorpus[docIndex])):
                    if textCorpus[docIndex][wordIndex]==i \
                            and tTilde[docIndex][wordIndex]==j:
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
            prob1 = computeRelativeProbabilityForTZ(activeTopics=samplingVariables.getActiveTopics(),
                                                    textCorpus=textCorpus, 
                                                    wordType=i, 
                                                    topicIndex=j, 
                                                    t=tTilde, 
                                                    z=zTilde,
                                                    gammas=samplingVariables.gammas, 
                                                    w=samplingVariables.w, 
                                                    alphaTheta=hyperParameters.alphaTheta, 
                                                    alphaF=hyperParameters.alphaF)
            prob2 = computeRelativeProbabilityForTZ(activeTopics=samplingVariables.getActiveTopics(),
                                                    textCorpus=textCorpus, 
                                                    wordType=i, 
                                                    topicIndex=j, 
                                                    t=samplingVariables.t, 
                                                    z=samplingVariables.z,
                                                    gammas=samplingVariables.gammas, 
                                                    w=samplingVariables.w, 
                                                    alphaTheta=hyperParameters.alphaTheta, 
                                                    alphaF=hyperParameters.alphaF)
            print "prob1:", prob1
            print "prob2:", prob2
            assert oneIfTopicAssignmentsSupported(textCorpus, samplingVariables.t, samplingVariables.z)==1
            assert oneIfTopicAssignmentsSupported(textCorpus, tTilde, zTilde)==1
            ratio = min(1.0, prob1/prob2)
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
        newTopicIndices = samplingVariables.createNewTopics(textCorpus, numNewTopics)
        for newTopicIndex in newTopicIndices:
            # fill the new Z row with all zeros, except for word i for which it should be 1
            for wordIndex in range(samplingVariables.z.shape[0]):
                samplingVariables.z[wordIndex,newTopicIndex] = 0
            samplingVariables.z[i,newTopicIndex] = 1

            # initialize new u column:
            for wordIndex in range(samplingVariables.z.shape[0]):
                samplingVariables.u[wordIndex,newTopicIndex] = 1.0
            samplingVariables.u[i,newTopicIndex] = \
                    prob.sampleRightTruncatedExponential(
                                         samplingVariables.gammas[i]*samplingVariables.w[j],
                                         1.0)
            
            # initialize new w value:
            gammaSum= sum([samplingVariables.gammas[i]*samplingVariables.u[i,j] \
                           for i in range(textCorpus.getVocabSize())])
            samplingVariables.w[newTopicIndex] = \
                    random.gammavariate(getNumWordTypesActivatedInTopic(j, samplingVariables.z) \
                                            -hyperParameters.sigma,
                                        1.0/(hyperParameters.tau+gammaSum)) 
            

def updateWGStar(textCorpus, samplingVariables, hyperParameters):
    # update w:
    for topicIndex in samplingVariables.getActiveTopicIndices(textCorpus, samplingVariables.t):
        gammaSum= sum([samplingVariables.gammas[i]*samplingVariables.u[i,topicIndex] \
               for i in range(textCorpus.getVocabSize())])
        samplingVariables.w[topicIndex] = \
                random.gammavariate(getNumWordTypesActivatedInTopic(topicIndex, samplingVariables.z) \
                                        -hyperParameters.sigma,
                                    1.0/(hyperParameters.tau+gammaSum)) 
    
    # update G*:
    # TODO: implement sampler for exponentially tilted distribution
    assert hyperParameters.sigma==0
    samplingVariables.gStar = random.gammavariate(
                                    hyperParameters.alpha,
                                    1.0/(hyperParameters.tau+sum(samplingVariables.gammas)))
    
def updateGammas(textCorpus, samplingVariables, hyperParameters):
    for wordType in textCorpus.getVocabSize():
        
        samplingVariables.gammas[wordType] = \
            np.random.gamma(hyperParameters.aGamma + samplingVariables[wordType,:].sum(),
                            hyperParameters.bGamma \
                                + sum([samplingVariables.w[j]*samplingVariables.u[wordType,j]\
                                       for j in samplingVariables.getActiveTopicIndices(textCorpus, 
                                                                      samplingVariables.t)]) \
                                + samplingVariables.gStar)
    
########################
### SAMPLING ###########
########################

def sampleTGivenZT(activeTopicIndices, samplingDoc, samplingWordPos, alphaTheta, alphaF, textCorpus, t, z,
                   excludeDocWords=[]):
    unnormalizedTopicProbs = []
    wordType = textCorpus[samplingDoc][samplingWordPos]
    for topic in activeTopicIndices:
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
                               for r in range(getNumWordTypesActivatedInTopic(topic, z))])
            unnormalizedTopicProbs.append(numerator1 * numerator2 / denominator)
    normalizer = sum(unnormalizedTopicProbs)
    normalizedTopicProbs = [p / normalizer for p in unnormalizedTopicProbs]
    return np.nonzero(np.random.multinomial(1, normalizedTopicProbs))[0][0]

def computeRelativeProbabilityForTZ(activeTopics, textCorpus, wordType, topicIndex, t, z, gammas, w, 
                                    alphaTheta, alphaF):
    if oneIfTopicAssignmentsSupported(textCorpus, t, z)!=1:
        return 0.0
    
    factor1 = (1.0 - math.exp(gammas[wordType]*w[topicIndex]))**z[topicIndex,wordType]
    
    factor2 = math.exp(-(1-z[topicIndex,wordType])*gammas[wordType]*w[topicIndex])
    
    factor3 = 1.0
    activeTopics = activeTopics
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
        subNumerator1 = np.gamma(getNumWordTypesActivatedInTopic(topic, z)*alphaF)
        subDenominator1 = np.gamma(alphaF) ** getNumWordTypesActivatedInTopic(topic, z)
        subNumerator2 = 1.0
        for r in range(getNumWordTypesActivatedInTopic(topic, z)):
            subNumerator2 *= np.gamma(alphaF + 
                                      getNumTopicAssignmentsToWordType(topic, 
                                                    getRthActiveWordTypeInTopic(r, topic, z), 
                                                    t))
        subDenominator2 = np.gamma(getNumWordTypesActivatedInTopic(topic, z)*alphaF \
                                   + sum([getRthActiveWordTypeInTopic(r, topic, z) \
                                          for r in range(getNumWordTypesActivatedInTopic(topic, z))]))
        factor4 *= subNumerator1 / subDenominator1 * subNumerator2 / subDenominator2
    return factor1 * factor2 * factor3 * factor4

def oneIfTopicAssignmentsSupported(textCorpus, t, z):
    for docIndex in range(len(t)):
        for wordPos in range(len(t[docIndex])):
            topic = t[docIndex][wordPos]
            wordType = textCorpus[docIndex][wordPos]
            if z[wordType, topic]!=1:
                return 0
    return 1

def sampleTruncatedNumNewTopics(activeTopicIndices, textCorpus, t, alphaTheta, wordType, gammas, alpha, sigma, tau, cutoff=20):
    
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

        