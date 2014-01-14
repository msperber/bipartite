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
    def __init__(self, alpha, sigma, tau, alphaTheta, alphaF, aGamma, bGamma):
        self.alphaTheta = alphaTheta
        self.alphaF = alphaF
        assert 0 <= sigma < 1.0
        self.alpha = alpha
        self.sigma = sigma
        self.tau = tau
        self.aGamma = aGamma
        self.bGamma = bGamma

class GibbsSamplingVariables(object):
    def __init__(self, textCorpus, nTopics = 1):
        self.deadTopics, self.activeTopics = [], []
        self.textCorpus = textCorpus
        self.allocateVars(textCorpus, nTopics)
        self.initWithFullTopicsAndGammasFromFrequencies(textCorpus, nTopics)
        
    def allocateVars(self, textCorpus, nTopics):
        vocabSize = textCorpus.getVocabSize()
        
        # scores for word-types in topics:
        self.uMat = np.empty((vocabSize, nTopics)) 
        
        # which word-types belong to which topics:
        self.zMat = np.empty((vocabSize, nTopics), dtype=np.int8)

        # topic assignments
        self.tLArr = []
        
        for doc in textCorpus:
            self.tLArr.append(np.empty(len(doc)))
            
        # reading interest ("word popularity")
        self.gammas = np.empty((vocabSize,))
        
        # topic popularity
        self.wArr = np.empty((nTopics,))
        
        self.gStar = None
        
        self.activeTopics = range(nTopics)
        
    def initWithFullTopicsAndGammasFromFrequencies(self, textCorpus, nTopics):
        # initialize variables to a consistent state
        # ensure cosistency by making all words belong to all topics initially
        self.uMat.fill(0.5)
        self.zMat.fill(1)
        for iteratingDoc in range(len(textCorpus)):
            self.tLArr[iteratingDoc] = np.random.randint(0, 
                                                         nTopics, 
                                                         len(self.tLArr[iteratingDoc]))
        wordFreqs = textCorpus.getVocabFrequencies()
        totalNumWords = textCorpus.getTotalNumWords()
        for iteratingWordType in range(textCorpus.getVocabSize()):
            self.gammas[iteratingWordType] = \
                        float(wordFreqs[iteratingWordType]) / float(totalNumWords) 
        self.wArr.fill(1.0)
    
    # approach to managing active & dead topics: both are stored in (complementary) lists,
    # which are only changed upon a call of releaseDeadTopics() or createNewTopics()
    # thus, topics with no associated words remain "active" until releaseDeadTopics() gets called
    
    def getActiveTopics(self):
        assert len(set(self.deadTopics).intersection(self.activeTopics))==0
        return self.activeTopics
    
    def releaseDeadTopics(self):
        assert len(set(self.deadTopics).intersection(self.activeTopics))==0
        removingTopics = []
        for topic in self.activeTopics:
            if getNumWordTypesActivatedInTopic(topic, self.zMat)==0:
                for i in range(len(self.tLArr)):
                    for j in range(self.tLArr[i].shape[0]):
                        assert self.tLArr[i][j]!=topic
                removingTopics.append(topic)
        for topic in removingTopics:
            self.activeTopics.remove(topic)
        self.deadTopics.extend(removingTopics)
        assert len(set(self.deadTopics).intersection(self.activeTopics))==0
        return removingTopics
    
    def createNewTopics(self, numNewTopics):
        assert len(set(self.deadTopics).intersection(self.activeTopics))==0
        # to be on the safe side: init new zMat's with 1, new uMat's with 0.5, new wArr with 1
        # TODO: make more efficient by grouping memory allocations
        newTopics = []
        for _ in range(numNewTopics):
            if len(self.deadTopics)==0:
                newTopic = len(self.activeTopics)
                # expand zMat
                newZ = np.ones((self.zMat.shape[0], self.zMat.shape[1]+1))
                newZ[:,:-1] = self.zMat
                self.zMat = newZ
                # expand uMat
                newU = np.empty((self.uMat.shape[0], self.uMat.shape[1]+1))
                for i in range(self.uMat.shape[0]):
                    newU[i,-1] = 0.5
                newU[:,:-1] = self.uMat
                self.uMat = newU
                # expand wArr
                newW = np.ones((self.wArr.shape[0]+1))
                newW[:-1] = self.wArr
                self.wArr = newW
            else:
                newTopic = self.deadTopics.pop()
                for i in range(self.zMat.shape[0]):
                    self.zMat[i,newTopic] = 1
                    self.uMat[i,newTopic] = 0.5
                self.wArr[newTopic] = 1.0
            newTopics.append(newTopic)
            self.activeTopics.append(newTopic)
            
        assert self.zMat.shape == self.uMat.shape
        assert len(set(self.deadTopics).intersection(self.activeTopics))==0
        return newTopics
    
    def removeTopic(self, topic):
        for i in range(len(self.tLArr)):
            for j in range(self.tLArr[i].shape[0]):
                assert self.tLArr[i][j]!=topic
        assert len(set(self.deadTopics).intersection(self.activeTopics))==0
        self.deadTopics.append(topic)
        self.activeTopics.remove(topic)
        assert len(set(self.deadTopics).intersection(self.activeTopics))==0

def getNumTopicOccurencesInDoc(topic, doc, tLArr,
                                    excludeDocWordPositions=[]):
    num = 0
    for iteratingWordPos in range(len(tLArr[doc])):
        if tLArr[doc][iteratingWordPos]==topic:
            if (doc, iteratingWordPos) not in excludeDocWordPositions:
                num += 1
    return num

def getNumTopicAssignmentsToWordType(topic, wordType, tLArr, textCorpus, 
                                    excludeDocWordPositions=[]):
    num = 0
    for iteratingDoc in range(len(tLArr)):
        for iteratingWordPos in range(len(tLArr[iteratingDoc])):
            if topic==tLArr[iteratingDoc][iteratingWordPos] \
                    and textCorpus[iteratingDoc][iteratingWordPos]==wordType:
                if (iteratingDoc, iteratingWordPos) not in excludeDocWordPositions:
                    num += 1
    return num

def getNumWordTypesActivatedInTopic(topic, zMat):
    return int(np.asscalar(zMat[:,topic].sum()))

def getRthActiveWordTypeInTopic(r, topic, zMat):
    return np.nonzero(zMat[:,topic])[0][r]

########################
### MAIN ALGORITHM #####
########################


def inferTopicsCollapsedGibbs(textCorpus, hyperParameters, numIterations):
    
    # initialize variables
    samplingVariables = GibbsSamplingVariables(textCorpus, nTopics = 20)
    
    for iteration in range(numIterations):
        print "Gibbs sampling iteration:", iteration
        print "gammas:", samplingVariables.gammas
        print "w's:", samplingVariables.wArr
        print "u's:", samplingVariables.uMat
        assert oneIfTopicAssignmentsSupported(textCorpus, samplingVariables.tLArr, 
                                              samplingVariables.zMat)==1
        updateUs(textCorpus=textCorpus, samplingVariables=samplingVariables)
        assert oneIfTopicAssignmentsSupported(textCorpus, samplingVariables.tLArr, 
                                              samplingVariables.zMat)==1
        updateZs(textCorpus, samplingVariables, hyperParameters)
        assert oneIfTopicAssignmentsSupported(textCorpus, samplingVariables.tLArr, 
                                              samplingVariables.zMat)==1
        updateWGStar(textCorpus, samplingVariables, hyperParameters)
        assert oneIfTopicAssignmentsSupported(textCorpus, samplingVariables.tLArr, 
                                              samplingVariables.zMat)==1
        updateGammas(textCorpus, samplingVariables, hyperParameters)
        
########################
### UPDATES ############
########################

def updateUs(textCorpus, samplingVariables):
    """
    follows [Caron, 2012, Section 5]
    """
    for iteratingWordType in range(textCorpus.getVocabSize()):
        for iteratingTopic in samplingVariables.getActiveTopics():
            if utility.approx_equal(samplingVariables.zMat[iteratingWordType][iteratingTopic], 0.0):
                samplingVariables.uMat[iteratingWordType][iteratingTopic] = 1.0
            else:
                samplingVariables.uMat[iteratingWordType][iteratingTopic] = \
                        prob.sampleRightTruncatedExponential(
                                         samplingVariables.gammas[iteratingWordType] * \
                                         samplingVariables.wArr[iteratingTopic],
                                         1.0)

def updateZs(textCorpus, samplingVariables, hyperParameters):
    """
    a Metropolis algorithm to update zMat's and tLArr's simultaneously 
    """
    for iteratingWordType in range(textCorpus.getVocabSize()):
        for iteratingTopic in samplingVariables.getActiveTopics():
            # skip the case where only topic j is active for word i: we need at
            # least one topic in which each word is activated
            if utility.approx_equal(samplingVariables.zMat[iteratingWordType,:].sum(), 1) \
                    and utility.approx_equal(
                            samplingVariables.zMat[iteratingWordType,iteratingTopic],1):
                continue
            
            assert getNumWordTypesActivatedInTopic(iteratingTopic, samplingVariables.zMat)>0
            
            
            LQij = []
            for iteratingDoc in range(len(textCorpus)):
                for iteratingWordPos in range(len(textCorpus[iteratingDoc])):
                    if textCorpus[iteratingDoc][iteratingWordPos]==iteratingWordType \
                            and samplingVariables.tLArr[iteratingDoc][iteratingWordPos]==iteratingTopic:
                        LQij.append((iteratingDoc, iteratingWordPos))
                        
            if utility.approx_equal(
                    getNumWordTypesActivatedInTopic(iteratingTopic, samplingVariables.zMat), 1) \
                    and utility.approx_equal(
                            samplingVariables.zMat[iteratingWordType,iteratingTopic],1):
                samplingVariables.removeTopic(iteratingTopic)
                for r in range(len(LQij)):
                    iteratingDoc, iteratingWordPos = LQij[r]
                    samplingVariables.tLArr[iteratingDoc][iteratingWordPos] = sampleTGivenZT(
                                activeTopics=samplingVariables.getActiveTopics(),
                                doc=iteratingDoc, 
                                wordPos=iteratingWordPos,
                                alphaTheta=hyperParameters.alphaTheta, 
                                alphaF=hyperParameters.alphaF,
                                textCorpus=textCorpus,
                                tLArr=samplingVariables.tLArr,
                                zMat=samplingVariables.zMat,
                                excludeDocWordPositions=LQij[r+1:])
            else:
                # switch z_ij between 0 and 1
                zTilde = samplingVariables.zMat.copy()
                zTilde[iteratingWordType,iteratingTopic] = 1 - zTilde[iteratingWordType,iteratingTopic]
                tTilde = copy.deepcopy(samplingVariables.tLArr)
                
                assert oneIfTopicAssignmentsSupported(textCorpus, samplingVariables.tLArr, 
                                                          samplingVariables.zMat) == 1

                # resample topics
                for r in range(len(LQij)):
                    iteratingDoc, iteratingWordPos = LQij[r]
                    tTilde[iteratingDoc][iteratingWordPos] = sampleTGivenZT(
                                activeTopics=samplingVariables.getActiveTopics(),
                                doc=iteratingDoc, 
                                wordPos=iteratingWordPos,
                                alphaTheta=hyperParameters.alphaTheta, 
                                alphaF=hyperParameters.alphaF,
                                textCorpus=textCorpus,
                                tLArr=tTilde,
                                zMat=zTilde,
                                excludeDocWordPositions=LQij[r+1:])
                
                assert oneIfTopicAssignmentsSupported(textCorpus, samplingVariables.tLArr, 
                                                          samplingVariables.zMat) == 1
                assert oneIfTopicAssignmentsSupported(textCorpus, tTilde, zTilde) == 1
                
                # compute relative probabilities
                if utility.approx_equal(
                        getNumWordTypesActivatedInTopic(iteratingTopic, samplingVariables.zMat), 1):
                    samplingVariables.zMat = zTilde
                    samplingVariables.tLArr = tTilde
                else:
                    prob1 = computeRelativeProbabilityForTZ(
                                    activeTopics=samplingVariables.getActiveTopics(),
                                    textCorpus=textCorpus, 
                                    wordType=iteratingWordType, 
                                    topic=iteratingTopic, 
                                    tLArr=tTilde, 
                                    zMat=zTilde,
                                    gammas=samplingVariables.gammas, 
                                    wArr=samplingVariables.wArr, 
                                    alphaTheta=hyperParameters.alphaTheta, 
                                    alphaF=hyperParameters.alphaF)
                    prob2 = computeRelativeProbabilityForTZ(
                                    activeTopics=samplingVariables.getActiveTopics(),
                                    textCorpus=textCorpus, 
                                    wordType=iteratingWordType, 
                                    topic=iteratingTopic, 
                                    tLArr=samplingVariables.tLArr, 
                                    zMat=samplingVariables.zMat,
                                    gammas=samplingVariables.gammas, 
                                    wArr=samplingVariables.wArr, 
                                    alphaTheta=hyperParameters.alphaTheta, 
                                    alphaF=hyperParameters.alphaF)
                    if prob2==0.0:
                        print "error: prob2 is 0! (prob2:", prob2, ")"
                        # TODO should this occur? numerical problem?
                        
    #                    prob2 = 0.0000001 
                        computeRelativeProbabilityForTZ(
                                    activeTopics=samplingVariables.getActiveTopics(),
                                    textCorpus=textCorpus, 
                                    wordType=iteratingWordType, 
                                    topic=iteratingTopic, 
                                    tLArr=samplingVariables.tLArr, 
                                    zMat=samplingVariables.zMat,
                                    gammas=samplingVariables.gammas, 
                                    wArr=samplingVariables.wArr, 
                                    alphaTheta=hyperParameters.alphaTheta, 
                                    alphaF=hyperParameters.alphaF)
                    ratio = min(1.0, prob1/prob2)
                    # accept or reject
                    if prob.flipCoin(ratio):
                        samplingVariables.zMat = zTilde
                        samplingVariables.tLArr = tTilde
        
        # remove dead topics
        nBefore = len(samplingVariables.getActiveTopics())
        assert oneIfTopicAssignmentsSupported(textCorpus, samplingVariables.tLArr, 
                                          samplingVariables.zMat) == 1

        assert nBefore==len(samplingVariables.getActiveTopics())

        # create new topics
        numNewTopics = sampleTruncatedNumNewTopics(
                                          activeTopics=samplingVariables.getActiveTopics(),
                                          textCorpus=textCorpus, 
                                          tLArr=samplingVariables.tLArr, 
                                          alphaTheta=hyperParameters.alphaTheta, 
                                          wordType=iteratingWordType,
                                          gammas=samplingVariables.gammas,
                                          alpha=hyperParameters.alpha, 
                                          sigma=hyperParameters.sigma, 
                                          tau=hyperParameters.tau)
        newTopics = samplingVariables.createNewTopics(numNewTopics)
        wordFreqs = textCorpus.getVocabFrequencies()
        totalNumWords = textCorpus.getTotalNumWords()
        assert oneIfTopicAssignmentsSupported(textCorpus, samplingVariables.tLArr, 
                                  samplingVariables.zMat) == 1

        for newTopic in newTopics:
            for i in range(len(samplingVariables.tLArr)):
                for j in range(samplingVariables.tLArr[i].shape[0]):
                    assert samplingVariables.tLArr[i][j]!=newTopic

            # TODO: how should these topics be properly initialized?
            samplingVariables.wArr[newTopic] = 1.0
            for iteratingWordType2 in range(samplingVariables.uMat.shape[0]):
                samplingVariables.gammas[iteratingWordType2] = \
                        float(wordFreqs[iteratingWordType2]) / float(totalNumWords) 
            
            # fill the new zMat row with all zeros, except for word i for which it should be 1
            assert oneIfTopicAssignmentsSupported(textCorpus, samplingVariables.tLArr, 
                                      samplingVariables.zMat) == 1

            for iteratingWordType2 in range(samplingVariables.zMat.shape[0]):
                samplingVariables.zMat[iteratingWordType2,newTopic] = 0
            samplingVariables.zMat[iteratingWordType,newTopic] = 1
            assert oneIfTopicAssignmentsSupported(textCorpus, samplingVariables.tLArr, 
                                      samplingVariables.zMat) == 1

            # initialize new uMat column:
            for iteratingWordType2 in range(samplingVariables.zMat.shape[0]):
                samplingVariables.uMat[iteratingWordType2,newTopic] = 1.0
            samplingVariables.uMat[iteratingWordType,newTopic] = \
                    prob.sampleRightTruncatedExponential(
                             samplingVariables.gammas[iteratingWordType] \
                                    * samplingVariables.wArr[newTopic],
                             1.0)
        assert oneIfTopicAssignmentsSupported(textCorpus, samplingVariables.tLArr, 
                                  samplingVariables.zMat) == 1

#            
#            # initialize new wArr value:
#            gammaSum= sum([samplingVariables.gammas[i]*samplingVariables.uMat[i,iteratingTopic] \
#                           for i in range(textCorpus.getVocabSize())])
#            samplingVariables.wArr[newTopic] = \
#                    random.gammavariate(getNumWordTypesActivatedInTopic(iteratingTopic, 
#                                                                        samplingVariables.zMat) \
#                                            -hyperParameters.sigma,
#                                        1.0/(hyperParameters.tau+gammaSum))
            

def updateWGStar(textCorpus, samplingVariables, hyperParameters):
    # update wArr:
    for iteratingTopic in samplingVariables.getActiveTopics():
        assert getNumWordTypesActivatedInTopic(iteratingTopic, samplingVariables.zMat)>0

        gammaSum= sum([samplingVariables.gammas[i]*samplingVariables.uMat[i,iteratingTopic] \
               for i in range(textCorpus.getVocabSize())])
        samplingVariables.wArr[iteratingTopic] = \
                random.gammavariate(getNumWordTypesActivatedInTopic(iteratingTopic, 
                                                                    samplingVariables.zMat) \
                                        - hyperParameters.sigma,
                                    1.0/(hyperParameters.tau+gammaSum)) 
    
    # update G*:
    # TODO: implement sampler for exponentially tilted distribution
    assert hyperParameters.sigma==0.0
    samplingVariables.gStar = random.gammavariate(
                                    hyperParameters.alpha,
                                    1.0/(hyperParameters.tau+sum(samplingVariables.gammas)))
    
def updateGammas(textCorpus, samplingVariables, hyperParameters):
    for iteratingWordType in range(textCorpus.getVocabSize()):
        
        samplingVariables.gammas[iteratingWordType] = \
            np.random.gamma(hyperParameters.aGamma \
                                + samplingVariables.zMat[iteratingWordType,:].sum(),
                            hyperParameters.bGamma \
                    + sum([samplingVariables.wArr[j]*samplingVariables.uMat[iteratingWordType,j]\
                           for j in samplingVariables.getActiveTopics()]) \
                    + samplingVariables.gStar)
    
########################
### SAMPLING ###########
########################

def sampleTGivenZT(activeTopics, doc, wordPos, alphaTheta, alphaF, textCorpus, tLArr, 
                   zMat, excludeDocWordPositions=[]):
    unnormalizedTopicProbs = []
    wordType = textCorpus[doc][wordPos]
    assert oneIfTopicAssignmentsSupported(textCorpus=textCorpus,
                                          tLArr=tLArr,
                                          zMat=zMat,
                                          excludeDocWordPositions=[(doc, wordPos)] \
                                                + excludeDocWordPositions) \
                == 1
    for iteratingTopic in activeTopics:
        if utility.approx_equal(zMat[wordType,iteratingTopic], 0):
            unnormalizedTopicProbs.append(0.0)
        else:
            numerator1 = alphaTheta + getNumTopicOccurencesInDoc(iteratingTopic, doc, tLArr,
                                    excludeDocWordPositions=[(doc,iteratingTopic)] + 
                                                            excludeDocWordPositions)
            numerator2 = np.random.gamma(alphaF + 
                                 getNumTopicAssignmentsToWordType(topic=iteratingTopic, 
                                                                  wordType=wordType, 
                                                                  tLArr=tLArr,
                                                                  textCorpus=textCorpus,
                                                                  excludeDocWordPositions=\
                                                                       [(doc,iteratingTopic)] + 
                                                                       excludeDocWordPositions))
            denominator = sum([alphaF + getNumTopicAssignmentsToWordType( \
                            topic=iteratingTopic,
                            wordType=getRthActiveWordTypeInTopic(r, iteratingTopic, zMat),
                            tLArr=tLArr, 
                            textCorpus=textCorpus, 
                            excludeDocWordPositions=[(doc,iteratingTopic)] + 
                                                    excludeDocWordPositions) \
                       for r in range(getNumWordTypesActivatedInTopic(iteratingTopic, zMat))])
            unnormalizedTopicProbs.append(numerator1 * numerator2 / denominator)
    normalizer = sum(unnormalizedTopicProbs)
    normalizedTopicProbs = [p / normalizer for p in unnormalizedTopicProbs]
    return activeTopics[np.nonzero(np.random.multinomial(1, normalizedTopicProbs))[0][0]]

def computeRelativeProbabilityForTZ(activeTopics, textCorpus, wordType, topic, tLArr, zMat, gammas, 
                                    wArr, alphaTheta, alphaF):
    if oneIfTopicAssignmentsSupported(textCorpus, tLArr, zMat)!=1:
        return 0.0
    
    factor1 = (1.0 - math.exp(-gammas[wordType]*wArr[topic]))**zMat[wordType,topic]
    
    factor2 = math.exp(-(1-zMat[wordType,topic])*gammas[wordType]*wArr[topic])
    
    factor3 = 1.0
    activeTopics = activeTopics
    for iteratingDoc in range(len(textCorpus)):
        subNumerator1 = np.random.gamma(len(activeTopics) * alphaTheta)
        subDenominator1 = np.random.gamma(alphaTheta) ** len(activeTopics)
        subNumerator2 = 1.0
        for iteratingTopic in activeTopics:
            subNumerator2 *= np.random.gamma(alphaTheta + 
                                             getNumTopicOccurencesInDoc(iteratingTopic, 
                                                                        iteratingDoc, 
                                                                        tLArr))
        subDenominator2 = np.random.gamma(len(activeTopics)*alphaTheta \
                                   + sum([getNumTopicOccurencesInDoc(iteratingTopic, 
                                                                     iteratingDoc, 
                                                                     tLArr) \
                                          for iteratingTopic in activeTopics]))
        factor3 *= subNumerator1 / subDenominator1 * subNumerator2 / subDenominator2
    
    factor4 = 1.0
    for iteratingTopic in activeTopics:
        if getNumWordTypesActivatedInTopic(iteratingTopic, zMat) == 0:
            continue
        subNumerator1 = np.random.gamma(
                                getNumWordTypesActivatedInTopic(iteratingTopic, zMat)*alphaF)
        subDenominator1 = np.random.gamma(alphaF) ** getNumWordTypesActivatedInTopic(iteratingTopic,
                                                                                     zMat)
        subNumerator2 = 1.0
        for r in range(getNumWordTypesActivatedInTopic(iteratingTopic, zMat)):
            subNumerator2 *= np.random.gamma(alphaF + 
                                      getNumTopicAssignmentsToWordType(
                                                    topic=iteratingTopic, 
                                                    wordType=getRthActiveWordTypeInTopic(
                                                                            r=r, 
                                                                            topic=iteratingTopic,
                                                                            zMat=zMat), 
                                                    tLArr=tLArr,
                                                    textCorpus=textCorpus))
        subDenominator2 = np.random.gamma(
                            getNumWordTypesActivatedInTopic(iteratingTopic, zMat)*alphaF \
                               + sum([getRthActiveWordTypeInTopic(r, iteratingTopic, zMat) \
                                      for r in range(getNumWordTypesActivatedInTopic(iteratingTopic,
                                                                                     zMat))]))
        factor4 *= subNumerator1 / subDenominator1 * subNumerator2 / subDenominator2
    return factor1 * factor2 * factor3 * factor4

def oneIfTopicAssignmentsSupported(textCorpus, tLArr, zMat, excludeDocWordPositions=[]):
    for iteratingDoc in range(len(tLArr)):
        for iteratingWordPos in range(len(tLArr[iteratingDoc])):
            if (iteratingDoc, iteratingWordPos) in excludeDocWordPositions:
                continue
            iteratingTopic = tLArr[iteratingDoc][iteratingWordPos]
            iteratingWordType = textCorpus[iteratingDoc][iteratingWordPos]
            if not utility.approx_equal(zMat[iteratingWordType, iteratingTopic], 1):
                return 0
    return 1

def sampleTruncatedNumNewTopics(activeTopics, textCorpus, tLArr, alphaTheta, wordType,
                                gammas, alpha, sigma, tau, cutoff=20):
    
    unnormalizedProbs = []
    for num in range(cutoff):
        factor1 = 1.0
        for iteratingDoc in range(len(textCorpus)):
            numerator = 1.0
            for j in activeTopics:
                numerator *= np.random.gamma(alphaTheta + \
                                      getNumTopicOccurencesInDoc(topic=j, 
                                                                 doc=iteratingDoc, 
                                                                 tLArr=tLArr))
            denominator = np.random.gamma(len(textCorpus[iteratingDoc]) + \
                                   (len(activeTopics)+ num) * alphaTheta )
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

        