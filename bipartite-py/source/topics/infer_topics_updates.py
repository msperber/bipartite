'''
Created on Jan 24, 2014

@author: Matthias Sperber
'''

import numpy as np
import source.utility as utility
import source.prob as prob
import math
import source.expressions as expr
import random
import copy
from infer_topics_state import *

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
            if topicExclusivelyActivatedForWord(wordType=iteratingWordType,
                                                topic=iteratingTopic, 
                                                samplingVariables=samplingVariables):
                continue

            # skip this topic if only one topic is active: there will be no valid proposals
            if len(samplingVariables.getActiveTopics()) == 1:
                break
            
            # core events take place here:
            proposeAndAcceptOrReject(iteratingWordType=iteratingWordType,
                                     iteratingTopic=iteratingTopic,
                                     samplingVariables=samplingVariables, 
                                     textCorpus=textCorpus, 
                                     hyperParameters=hyperParameters)

        
        # remove dead topics
        samplingVariables.releaseDeadTopics()

        # create new topics
        createNewTopics(iteratingWordType=iteratingWordType,
                        textCorpus= textCorpus,
                        samplingVariables=samplingVariables,
                        hyperParameters=hyperParameters)

def topicExclusivelyActivatedForWord(wordType, topic, samplingVariables):
    return getNumActiveTopicsForWordType(wordType, 
                                 samplingVariables.zMat, 
                                 samplingVariables.getActiveTopics()) == 1 \
        and utility.approx_equal(
                samplingVariables.zMat[wordType,topic],1)
        
def proposeAndAcceptOrReject(iteratingWordType, iteratingTopic, samplingVariables, 
                             textCorpus, hyperParameters):
    LQij = []
    for iteratingDoc in range(len(textCorpus)):
        for iteratingWordPos in range(len(textCorpus[iteratingDoc])):
            if textCorpus[iteratingDoc][iteratingWordPos]==iteratingWordType \
                    and samplingVariables.tLArr[iteratingDoc][iteratingWordPos]==iteratingTopic:
                LQij.append((iteratingDoc, iteratingWordPos))

    # switch z_ij between 0 and 1
    zTilde = samplingVariables.zMat.copy()
    zTilde[iteratingWordType, iteratingTopic] = 1 - zTilde[iteratingWordType, iteratingTopic]
    tTilde = copy.deepcopy(samplingVariables.tLArr)
    
    # resample topics
    # careful: these are changed in-place (don't want to re-allocate a new array every
    # single step), so must be changed back immediately after
    numWordTypesActivatedInTopics = samplingVariables.counts.numWordTypesActivatedInTopic
    if zTilde[iteratingWordType, iteratingTopic]==1:
        numWordTypesActivatedInTopics[iteratingTopic] =\
                numWordTypesActivatedInTopics.get(iteratingTopic,0) + 1
    else: 
        numWordTypesActivatedInTopics[iteratingTopic] =\
                numWordTypesActivatedInTopics.get(iteratingTopic,0) - 1
    # store delta counts so we can revert the changes if the proposal is rejected
    numTopicAssignmentsToWordTypeDeltaTilde = {}
    numTopicOccurencesInDocTilde = \
            copy.deepcopy(samplingVariables.counts.numTopicOccurencesInDoc)
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
                    excludeDocWordPositions=LQij[r+1:],
                    numWordTypesActivatedInTopics=numWordTypesActivatedInTopics,
                    numTopicAssignmentsToWordType=\
                                samplingVariables.counts.numTopicAssignmentsToWordType)
        # update numTopicAssignmentsToWordType counts
        samplingVariables.counts.numTopicAssignmentsToWordType[textCorpus[iteratingDoc]\
                            [iteratingWordPos],tTilde[iteratingDoc][iteratingWordPos]] \
                = samplingVariables.counts.numTopicAssignmentsToWordType.get( \
                    (textCorpus[iteratingDoc][iteratingWordPos],
                     tTilde[iteratingDoc][iteratingWordPos]), 0) + 1
        samplingVariables.counts.numTopicAssignmentsToWordType[textCorpus[iteratingDoc]\
            [iteratingWordPos],samplingVariables.tLArr[iteratingDoc][iteratingWordPos]] \
                -= 1
        numTopicAssignmentsToWordTypeDeltaTilde[textCorpus[iteratingDoc]\
                            [iteratingWordPos],tTilde[iteratingDoc][iteratingWordPos]] \
            = numTopicAssignmentsToWordTypeDeltaTilde.get((textCorpus[iteratingDoc]\
                        [iteratingWordPos],tTilde[iteratingDoc][iteratingWordPos]), 0) + 1
        numTopicAssignmentsToWordTypeDeltaTilde[textCorpus[iteratingDoc]\
                            [iteratingWordPos],samplingVariables.tLArr[iteratingDoc][iteratingWordPos]] \
            = numTopicAssignmentsToWordTypeDeltaTilde.get((textCorpus[iteratingDoc]\
        [iteratingWordPos],samplingVariables.tLArr[iteratingDoc][iteratingWordPos]), 0) - 1

        # update numTopicOccurencesInDoc counts
        numTopicOccurencesInDocTilde[iteratingDoc,
                                                tTilde[iteratingDoc][iteratingWordPos]] = \
                numTopicOccurencesInDocTilde.get((iteratingDoc,
                                            tTilde[iteratingDoc][iteratingWordPos]),0) + 1
        numTopicOccurencesInDocTilde[iteratingDoc,
                                samplingVariables.tLArr[iteratingDoc][iteratingWordPos]] = \
                numTopicOccurencesInDocTilde.get((iteratingDoc,
                            samplingVariables.tLArr[iteratingDoc][iteratingWordPos]),0) - 1

    # compute relative probabilities
    
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
                    alphaF=hyperParameters.alphaF,
                    numWordTypesActivatedInTopics=numWordTypesActivatedInTopics,
                    numTopicOccurencesInDoc=numTopicOccurencesInDocTilde)
    # change back to the original state
    if zTilde[iteratingWordType, iteratingTopic]==1:
        numWordTypesActivatedInTopics[iteratingTopic] =\
                numWordTypesActivatedInTopics.get(iteratingTopic,0) - 1
    else: 
        numWordTypesActivatedInTopics[iteratingTopic] =\
                numWordTypesActivatedInTopics.get(iteratingTopic,0) + 1
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
                    alphaF=hyperParameters.alphaF,
                    numWordTypesActivatedInTopics=numWordTypesActivatedInTopics,
                    numTopicOccurencesInDoc=samplingVariables.counts.numTopicOccurencesInDoc)
    ratio = min(1.0, prob1/prob2)
    # accept or reject
    if prob.flipCoin(ratio):
        samplingVariables.zMat = zTilde
        samplingVariables.tLArr = tTilde
        samplingVariables.counts.numTopicOccurencesInDoc = numTopicOccurencesInDocTilde
        if samplingVariables.zMat[iteratingWordType, iteratingTopic]==1:
            samplingVariables.counts.numActiveTopicsForWordType[iteratingWordType] += 1
            samplingVariables.counts.numWordTypesActivatedInTopic[iteratingTopic] += 1
        else:
            samplingVariables.counts.numActiveTopicsForWordType[iteratingWordType] -= 1
            samplingVariables.counts.numWordTypesActivatedInTopic[iteratingTopic] -= 1
    else:
        # revert changes to samplingVariables.counts.numTopicAssignmentsToWordType:
        for (i,j) in numTopicAssignmentsToWordTypeDeltaTilde:
            samplingVariables.counts.numTopicAssignmentsToWordType[i,j] -= \
                    numTopicAssignmentsToWordTypeDeltaTilde[i,j]

def createNewTopics(iteratingWordType, textCorpus, samplingVariables, hyperParameters):
    numNewTopics = sampleTruncatedNumNewTopics(
                                  activeTopics=samplingVariables.getActiveTopics(),
                                  textCorpus=textCorpus, 
                                  tLArr=samplingVariables.tLArr, 
                                  alphaTheta=hyperParameters.alphaTheta, 
                                  wordType=iteratingWordType,
                                  gammas=samplingVariables.gammas,
                                  alpha=hyperParameters.alpha, 
                                  sigma=hyperParameters.sigma, 
                                  tau=hyperParameters.tau,
                                  numTopicOccurencesInDoc=\
                                        samplingVariables.counts.numTopicOccurencesInDoc)
    
    newTopics = samplingVariables.createNewTopics(numNewTopics)
    samplingVariables.counts.numActiveTopicsForWordType[iteratingWordType] += numNewTopics
    print "nr new topics", numNewTopics
    wordFreqs = textCorpus.getVocabFrequencies()
    totalNumWords = textCorpus.getTotalNumWords()

    for newTopic in newTopics:
        samplingVariables.counts.numWordTypesActivatedInTopic[newTopic] = \
                samplingVariables.counts.numWordTypesActivatedInTopic.get(newTopic,0) + 1

        samplingVariables.wArr[newTopic] = 1.0
        for iteratingWordType2 in range(samplingVariables.uMat.shape[0]):
            samplingVariables.gammas[iteratingWordType2] = \
                    float(wordFreqs[iteratingWordType2]) / float(totalNumWords) 
        
        # fill the new zMat row with all zeros, except for word i for which it should be 1

        for iteratingWordType2 in range(samplingVariables.zMat.shape[0]):
            samplingVariables.zMat[iteratingWordType2,newTopic] = 0
        samplingVariables.zMat[iteratingWordType,newTopic] = 1

        # initialize new uMat column:
        for iteratingWordType2 in range(samplingVariables.zMat.shape[0]):
            samplingVariables.uMat[iteratingWordType2,newTopic] = 1.0
        samplingVariables.uMat[iteratingWordType,newTopic] = \
                prob.sampleRightTruncatedExponential(
                         samplingVariables.gammas[iteratingWordType] \
                                * samplingVariables.wArr[newTopic],
                         1.0)


def updateWGStar(textCorpus, samplingVariables, hyperParameters):
    # update wArr:
    for iteratingTopic in samplingVariables.getActiveTopics():

        gammaSum= sum([samplingVariables.gammas[i]*samplingVariables.uMat[i,iteratingTopic] \
               for i in range(textCorpus.getVocabSize())])
        samplingVariables.wArr[iteratingTopic] = \
                np.random.gamma(samplingVariables.counts.numWordTypesActivatedInTopic.get( \
                                iteratingTopic, 0) \
                                        - hyperParameters.sigma,
                                    1.0/(hyperParameters.tau+gammaSum)) 
    
    # update G*:
    # TODO: implement sampler for exponentially tilted distribution
    assert hyperParameters.sigma==0.0
    samplingVariables.gStar = np.random.gamma(
                                    hyperParameters.alpha,
                                    1.0/(hyperParameters.tau+sum(samplingVariables.gammas)))
    
def updateGammas(textCorpus, samplingVariables, hyperParameters):
    for iteratingWordType in range(textCorpus.getVocabSize()):
        
        samplingVariables.gammas[iteratingWordType] = \
            np.random.gamma(hyperParameters.aGamma \
                                + samplingVariables.zMat[iteratingWordType,:].sum(),
                            1.0/(hyperParameters.bGamma \
                    + sum([samplingVariables.wArr[j]*samplingVariables.uMat[iteratingWordType,j]\
                           for j in samplingVariables.getActiveTopics()]) \
                    + samplingVariables.gStar))
    
def sampleTGivenZT(activeTopics, doc, wordPos, alphaTheta, alphaF, textCorpus, tLArr, 
                   zMat, numWordTypesActivatedInTopics, numTopicAssignmentsToWordType,
                   excludeDocWordPositions=[]):
    unnormalizedTopicProbs = []
    wordType = textCorpus[doc][wordPos]
    for iteratingTopic in activeTopics:
        numTopicAssignmentsToWordTypeCount = GibbsCounts.getNumTopicAssignmentsToWordTypeExcl(\
                                    wordType=wordType, topic=iteratingTopic, tLArr=tLArr,
                                    textCorpus=textCorpus, 
                                    numTopicAssignmentsToWordTypeDict=numTopicAssignmentsToWordType,
                                    excludeDocWordPositions=[(doc,wordPos)] + excludeDocWordPositions)
        if utility.approx_equal(zMat[wordType,iteratingTopic], 0):
            unnormalizedTopicProbs.append(0.0)
        else:
            numerator1 = alphaTheta + numTopicAssignmentsToWordTypeCount
            numerator2 = math.gamma(alphaF + numTopicAssignmentsToWordTypeCount)
            denominator = sum([alphaF + GibbsCounts.getNumTopicAssignmentsToWordTypeExcl(\
                                        wordType=getRthActiveWordTypeInTopic(r, iteratingTopic, zMat),
                                        topic=iteratingTopic, tLArr=tLArr,
                                        textCorpus=textCorpus, 
                                        numTopicAssignmentsToWordTypeDict=numTopicAssignmentsToWordType,
                                        excludeDocWordPositions=[(doc,wordPos)] + excludeDocWordPositions) \
                       for r in range(numWordTypesActivatedInTopics.get(iteratingTopic,0))])

            unnormalizedTopicProbs.append(numerator1 * numerator2 / denominator)
    normalizer = sum(unnormalizedTopicProbs)
    normalizedTopicProbs = [p / normalizer for p in unnormalizedTopicProbs]
    return activeTopics[np.nonzero(np.random.multinomial(1, normalizedTopicProbs))[0][0]]

def computeRelativeProbabilityForTZ(activeTopics, textCorpus, wordType, topic, tLArr, zMat, gammas, 
                wArr, alphaTheta, alphaF, numWordTypesActivatedInTopics, numTopicOccurencesInDoc):
    if oneIfTopicAssignmentsSupported(textCorpus, tLArr, zMat)!=1:
        return 0.0
    
    factor1 = (1.0 - math.exp(-gammas[wordType]*wArr[topic]))**zMat[wordType,topic]
    
    factor2 = math.exp(-(1-zMat[wordType,topic])*gammas[wordType]*wArr[topic])
    
    factor3 = 1.0
    activeTopics = activeTopics
    for iteratingDoc in range(len(textCorpus)):
        subNumerator1 = math.gamma(len(activeTopics) * alphaTheta)
        subDenominator1 = math.gamma(alphaTheta) ** len(activeTopics)
        subNumerator2 = 1.0
        for iteratingTopic in activeTopics:
            subNumerator2 *= math.gamma(alphaTheta + 
                                     numTopicOccurencesInDoc.get((iteratingDoc,iteratingTopic),0))
        subDenominator2 = math.gamma(len(activeTopics)*alphaTheta \
                           + sum([numTopicOccurencesInDoc.get((iteratingDoc, iteratingTopic), 0) \
                                          for iteratingTopic in activeTopics]))
        factor3 *= subNumerator1 / subDenominator1 * subNumerator2 / subDenominator2
    
    factor4 = 1.0
    for iteratingTopic in activeTopics:
        if numWordTypesActivatedInTopics.get(iteratingTopic, 0) == 0:
            continue
        subNumerator1 = math.gamma(
                                numWordTypesActivatedInTopics.get(iteratingTopic, 0)*alphaF)
        subDenominator1 = math.gamma(alphaF) ** numWordTypesActivatedInTopics.get(iteratingTopic,
                                                                                     0)
        subNumerator2 = 1.0
        for r in range(numWordTypesActivatedInTopics.get(iteratingTopic, 0)):
            subNumerator2 *= math.gamma(alphaF + 
                                      getNumTopicAssignmentsToWordType(
                                                    topic=iteratingTopic, 
                                                    wordType=getRthActiveWordTypeInTopic(
                                                                            r=r, 
                                                                            topic=iteratingTopic,
                                                                            zMat=zMat), 
                                                    tLArr=tLArr,
                                                    textCorpus=textCorpus))
        subDenominator2 = math.gamma(
                            numWordTypesActivatedInTopics.get(iteratingTopic, 0)*alphaF \
                               + sum([getRthActiveWordTypeInTopic(r, iteratingTopic, zMat) \
                                      for r in range(numWordTypesActivatedInTopics.get(iteratingTopic,
                                                                                     0))]))
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
                                gammas, alpha, sigma, tau, numTopicOccurencesInDoc, cutoff=30):
    
    unnormalizedProbs = []
    for kiPlus in range(cutoff):
        factor1 = 1.0
        for iteratingDoc in range(len(textCorpus)):
            numerator = math.gamma((len(activeTopics)+kiPlus) * alphaTheta)
            for j in activeTopics:
                numerator *= math.gamma(alphaTheta + \
                                      numTopicOccurencesInDoc.get((iteratingDoc, j),0))
            denominator = math.gamma(len(textCorpus[iteratingDoc]) + \
                                   (len(activeTopics)+ kiPlus) * alphaTheta )
            factor1 *= numerator / denominator
        lamPoisson = expr.psiTildeFunction(t=gammas[wordType], 
                                    b=sum(gammas)-gammas[wordType],
                                    alpha=alpha,
                                    sigma=sigma,
                                    tau=tau)
        factor2 = lamPoisson**kiPlus * math.exp(-lamPoisson) / math.factorial(kiPlus)
        unnormalizedProbs.append(factor1 * factor2)
    normalizer = sum(unnormalizedProbs)
    normalizedProbs = [p / normalizer for p in unnormalizedProbs]
    return np.nonzero(np.random.multinomial(1, normalizedProbs))[0][0]
