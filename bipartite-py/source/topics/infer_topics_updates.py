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
from source.exptiltedstable import *
from scipy.special import gammaln

from source.topics.state import *
from source.expressions import psiTildeFunction, kappaFunction, psiFunction
from numpy.ma.testutils import assert_almost_equal

########################
### UPDATES ############
########################

def updateUs(textCorpus, samplingVariables):
    """
    follows [Caron, 2012, Section 5]
    """
    for iteratingWordType in range(textCorpus.getVocabSize()):
        for iteratingTopic in samplingVariables.getActiveTopics():
            if abs(samplingVariables.zMat[iteratingWordType][iteratingTopic] - 0.0) <= 1e-6:
                samplingVariables.uMat[iteratingWordType][iteratingTopic] = 1.0
            else:
                samplingVariables.uMat[iteratingWordType][iteratingTopic] = \
                        prob.sampleRightTruncatedExponential(
                                         samplingVariables.gammas[iteratingWordType] * \
                                         samplingVariables.wArr[iteratingTopic],
                                         1.0)


def updateTs(textCorpus, samplingVariables, hyperParameters):
    for iteratingDocument in range(len(textCorpus)):
        for iteratingWordPos in range(len(textCorpus[iteratingDocument])):
            currentWordType = textCorpus[iteratingDocument][iteratingWordPos]
            prevTopic = samplingVariables.tLArr[iteratingDocument][iteratingWordPos]
            newTopic, _ = \
                    sampleTGivenZT(activeTopics=samplingVariables.getActiveTopics(),
                            doc=iteratingDocument,
                            wordPos=iteratingWordPos, 
                            alphaTheta=hyperParameters.alphaTheta,
                            alphaF=hyperParameters.alphaF, 
                            textCorpus=textCorpus, 
                            tLArr=samplingVariables.tLArr, 
                            zMat=samplingVariables.zMat,
                            numTopicOccurencesInDoc=samplingVariables.counts.numTopicOccurencesInDoc,
                            numWordTypesActivatedInTopics=samplingVariables.counts.numWordTypesActivatedInTopic, 
                            numTopicAssignmentsToWordType=samplingVariables.counts.numTopicAssignmentsToWordType, 
                            excludeDocWordPositions=[(iteratingDocument, iteratingWordPos)])
            samplingVariables.tLArr[iteratingDocument][iteratingWordPos] = newTopic
            samplingVariables.counts.updateChangeInT(docPos=iteratingDocument, 
                                                     wordPos=iteratingWordPos, 
                                                     wordType=currentWordType, 
                                                     oldTopic=prevTopic, 
                                                     newTopic=newTopic)
#            samplingVariables.counts.docWordPosListForTopicAssignments[currentWordType,prevTopic].remove((iteratingDocument,iteratingWordPos))
#            if (currentWordType,newTopic) not in samplingVariables.counts.docWordPosListForTopicAssignments:
#                samplingVariables.counts.docWordPosListForTopicAssignments[currentWordType,newTopic] = []
#            samplingVariables.counts.docWordPosListForTopicAssignments[currentWordType,newTopic].append((iteratingDocument,iteratingWordPos))
#            samplingVariables.counts.numTopicAssignmentsToWordType[ \
#                                    currentWordType,
#                                    prevTopic] \
#                            = samplingVariables.counts.numTopicAssignmentsToWordType.get(( \
#                                    currentWordType,
#                                    prevTopic),0) - 1
#            samplingVariables.counts.numTopicAssignmentsToWordType[ \
#                                currentWordType,
#                                newTopic] \
#                        = samplingVariables.counts.numTopicAssignmentsToWordType.get(( \
#                                currentWordType,
#                                newTopic), 0) + 1
#            samplingVariables.counts.numTopicOccurencesInDoc[iteratingDocument,prevTopic] = \
#                    samplingVariables.counts.numTopicOccurencesInDoc.get((iteratingDocument,prevTopic),0) \
#                    -1
#            samplingVariables.counts.numTopicOccurencesInDoc[iteratingDocument,newTopic] = \
#                    samplingVariables.counts.numTopicOccurencesInDoc.get((iteratingDocument,newTopic),0) \
#                    +1

    
def updateWGStar(textCorpus, samplingVariables, hyperParameters):
    # update wArr:
    if samplingVariables.wArr.shape[0] < samplingVariables.zMat.shape[1]:
        samplingVariables.wArr = np.empty((samplingVariables.zMat.shape[1],))
    for iteratingTopic in samplingVariables.getActiveTopics():

        gammaSum= sum([samplingVariables.gammas[i]*samplingVariables.uMat[i,iteratingTopic] \
               for i in range(textCorpus.getVocabSize())])
        samplingVariables.wArr[iteratingTopic] = \
                np.random.gamma(samplingVariables.counts.numWordTypesActivatedInTopic.get( \
                                iteratingTopic, 0) \
                                        - hyperParameters.sigma,
                                    1.0/(hyperParameters.tau+gammaSum)) 
    

    if (abs( hyperParameters.sigma - 0) <= 1e-6):
        samplingVariables.gStar = np.random.gamma(
                                    hyperParameters.alpha,
                                    1.0/(hyperParameters.tau+sum(samplingVariables.gammas)))
    else :
        samplingVariables.gStar = establernd(hyperParameters.alpha/hyperParameters.sigma,hyperParameters.sigma,
                                             1.0/(hyperParameters.tau+sum(samplingVariables.gammas)),1)[0]
        
    
def updateGammas(textCorpus, samplingVariables, hyperParameters):
    for iteratingWordType in range(textCorpus.getVocabSize()):
        
        samplingVariables.gammas[iteratingWordType] = \
            np.random.gamma(hyperParameters.aGamma \
                                + samplingVariables.zMat[iteratingWordType,:].sum(),
                            1.0/(hyperParameters.bGamma \
                    + sum([samplingVariables.wArr[j]*samplingVariables.uMat[iteratingWordType,j]\
                           for j in samplingVariables.getActiveTopics()]) \
                    + samplingVariables.gStar))
    
def probDistributionTGivenZT(activeTopics, doc, wordPos, alphaTheta, alphaF, textCorpus, tLArr, 
                   zMat, numWordTypesActivatedInTopics, numTopicAssignmentsToWordType,
                   numTopicOccurencesInDoc,
                   c_theta = expr.c_theta_K,
                   c_f = expr.c_f_mj,
                   excludeDocWordPositions=[]):
    unnormalizedTopicProbs1, unnormalizedTopicProbs2 = [], []
    wordType = textCorpus[doc][wordPos]
    topicOfCurrentWord = tLArr[doc][wordPos]
    for iteratingTopic in activeTopics:
        if zMat[wordType,iteratingTopic] <= 1e-6:
            unnormalizedTopicProbs1.append(0.0)
            unnormalizedTopicProbs2.append(0.0)
        else:
            numTopicAssignmentsToDocCount = numTopicOccurencesInDoc.get((doc, iteratingTopic), 0)
            if iteratingTopic==topicOfCurrentWord: numTopicAssignmentsToDocCount -= 1
            numTopicAssignmentsToWordTypeCount = GibbsCounts.getNumTopicAssignmentsToWordTypeExcl(\
                                        wordType=wordType, topic=iteratingTopic, tLArr=tLArr,
                                        textCorpus=textCorpus, 
                                        numTopicAssignmentsToWordTypeDict=numTopicAssignmentsToWordType,
                                        excludeDocWordPositions=[(doc,wordPos)] + excludeDocWordPositions)
            summand1 = math.log(alphaTheta/c_theta(len(activeTopics)) + numTopicAssignmentsToDocCount)
            summand2 = math.log(alphaF/c_f(numWordTypesActivatedInTopics[iteratingTopic]) + numTopicAssignmentsToWordTypeCount)
#            summand2 = gammaln(alphaF/c_f(numWordTypesActivatedInTopics[iteratingTopic]) + numTopicAssignmentsToWordTypeCount + 1.0)
            summand3 = -math.log(alphaF*numWordTypesActivatedInTopics[iteratingTopic]/c_f(numWordTypesActivatedInTopics[iteratingTopic]) \
                                 + sum([GibbsCounts.getNumTopicAssignmentsToWordTypeExcl(\
                                        wordType=getRthActiveWordTypeInTopic(r, iteratingTopic, zMat),
                                        topic=iteratingTopic, tLArr=tLArr,
                                        textCorpus=textCorpus, 
                                        numTopicAssignmentsToWordTypeDict=numTopicAssignmentsToWordType,
                                        excludeDocWordPositions=[(doc,wordPos)] + excludeDocWordPositions) \
                       for r in range(numWordTypesActivatedInTopics.get(iteratingTopic,0))]))
            propProb2 = math.exp(summand1 + summand2 + summand3)
            unnormalizedTopicProbs2.append(propProb2)
    normalizer2 = sum(unnormalizedTopicProbs2)
    normalizedTopicProbs2 = [p / normalizer2 for p in unnormalizedTopicProbs2]
    return normalizedTopicProbs2
    
def sampleTGivenZT(activeTopics, doc, wordPos, alphaTheta, alphaF, textCorpus, tLArr, 
                   zMat, numWordTypesActivatedInTopics, numTopicAssignmentsToWordType,
                   numTopicOccurencesInDoc,
                   c_theta = expr.c_theta_K,
                   c_f = expr.c_f_mj,
                   excludeDocWordPositions=[]):
    normalizedTopicProbs = probDistributionTGivenZT(activeTopics, doc, wordPos, alphaTheta, alphaF, textCorpus, tLArr, 
                   zMat, numWordTypesActivatedInTopics, numTopicAssignmentsToWordType,
                   numTopicOccurencesInDoc,
                   c_theta = expr.c_theta_K,
                   c_f = expr.c_f_mj,
                   excludeDocWordPositions=[])
    rawTopic = np.nonzero(np.random.multinomial(1, normalizedTopicProbs))[0][0]
    topic = activeTopics[rawTopic]
    logProb = math.log(normalizedTopicProbs[rawTopic])
    return (topic, logProb)

#def computeRelativeLogProbabilityForTZOld(activeTopics, textCorpus, wordType, topic, tLArr, zMat, 
#                                       gammas, wArr, alphaTheta, alphaF, 
#                                       numWordTypesActivatedInTopics, numTopicOccurencesInDoc,
#                                       numTopicAssignmentsToWordType,
#                                       c_theta = expr.c_theta_K,
#                                       c_f = expr.c_f_mj,
#                                       topicsMightDie=True):
#    
#    # this check seems useless because it can never get false, but takes up quite some time..
#    if oneIfTopicAssignmentsSupported(textCorpus, tLArr, zMat)!=1:
#        return float("-inf")
#    
#    summand1 = zMat[wordType,topic] * math.log(1.0 - math.exp(-gammas[wordType]*wArr[topic]))
#    
#    summand2 = -(1-zMat[wordType,topic])*gammas[wordType]*wArr[topic]
#    
#    
#    # p(t,wo | z)
#    
#    summand3 = 0.0
#    summand4 = 0.0
#    if topicsMightDie:
#        for iteratingDoc in range(len(textCorpus)):
#            summand3 += gammaln(len(activeTopics)/c_theta(len(activeTopics)) * alphaTheta)
#        
#        summand4 += -len(activeTopics) * gammaln(alphaTheta/c_theta(len(activeTopics)))
#     
#    summand5, summand6 = 0.0, 0.0 
#    for iteratingDoc in range(len(textCorpus)):
#        for iteratingTopic in activeTopics:
#            summand5 += gammaln(alphaTheta/c_theta(len(activeTopics)) + \
#                                numTopicOccurencesInDoc.get((iteratingDoc, iteratingTopic), 0.0))
#        summand6 -= gammaln(len(activeTopics)/c_theta(len(activeTopics))*alphaTheta + \
#                    sum([numTopicOccurencesInDoc.get((iteratingDoc, j),0.0) for j in activeTopics]))
#    
#    if numWordTypesActivatedInTopics[topic] > 0:
#        summand7 = gammaln(numWordTypesActivatedInTopics[topic] \
#                           /  c_f(numWordTypesActivatedInTopics[topic])) * alphaF
#        summand8 = -numWordTypesActivatedInTopics[topic] \
#                        * gammaln(alphaF / c_f(numWordTypesActivatedInTopics[topic]))
#    else:
#        summand7, summand8 = 0, 0
#        
#    summand9 = 0.0
#    for iteratingTopic in activeTopics:
#        numWordsAssignedToTopic = 0.0
#        for r in range(numWordTypesActivatedInTopics[iteratingTopic]):
#            numWordsAssignedToTopic = numTopicAssignmentsToWordType[(iteratingTopic, 
#                                                               getRthActiveWordTypeInTopic(
#                                                                            r=r, 
#                                                                            topic=iteratingTopic,
#                                                                            zMat=zMat))]
#
#        for r in range(numWordTypesActivatedInTopics[iteratingTopic]):
#            topicsForWord = numTopicAssignmentsToWordType[(iteratingTopic, 
#                                                               getRthActiveWordTypeInTopic(
#                                                                            r=r, 
#                                                                            topic=iteratingTopic,
#                                                                            zMat=zMat))]
#            summand9 += gammaln(alphaF/c_f(numWordTypesActivatedInTopics[iteratingTopic])+topicsForWord)
#            
#        summand9 += -gammaln(numWordTypesActivatedInTopics[iteratingTopic] \
#                            /c_f(numWordTypesActivatedInTopics[iteratingTopic]) + numWordsAssignedToTopic)
#    
#    return summand1 + summand2 + summand3 + summand4 + summand5 + summand6 + summand7 + summand8 + summand9


def oneIfTopicAssignmentsSupported(textCorpus, tLArr, zMat, excludeDocWordPositions=[]):
    for iteratingDoc in range(len(tLArr)):
        for iteratingWordPos in range(len(tLArr[iteratingDoc])):
            if (iteratingDoc, iteratingWordPos) in excludeDocWordPositions:
                continue
            iteratingTopic = tLArr[iteratingDoc][iteratingWordPos]
            iteratingWordType = textCorpus[iteratingDoc][iteratingWordPos]
            if abs(zMat[iteratingWordType, iteratingTopic] - 1) > 0.1:
                return 0
    return 1



# Computes the log Equation (11) in the paper
def computeLMarginDistribution(textCorpus, gammas, zMat, uMat, activeTopics, alpha, sigma, tau):
    
    factor1 = 1.0
    for iteratingWordType in range(textCorpus.getVocabSize()):
        factor1 *= gammas[iteratingWordType] \
                ** getNumActiveTopicsForWordType(iteratingWordType, zMat, activeTopics)
                
    factor2 = math.exp(-psiFunction(sum(gammas), alpha, sigma, tau))
    
    factor3 = 1.0
    for iteratingTopic in activeTopics:
        sumGammaU = 0.0
        for iteratingWordType in range(textCorpus.getVocabSize()):
            sumGammaU += gammas[iteratingWordType] * uMat[iteratingWordType][iteratingTopic]
        factor3 *= kappaFunction(getNumWordTypesActivatedInTopic(iteratingTopic, zMat),
                                                 sumGammaU,
                                                 alpha, sigma, 
                                                 tau)
    return factor1 * factor2 * factor3

def computeLogLikelihoodTWZ(activeTopics, textCorpus, tLArr, zMat, 
                alphaTheta, alphaF, numWordTypesActivatedInTopics, 
                numTopicOccurencesInDoc, c_theta=expr.c_theta_K, c_f=expr.c_f_mj):
    
    # this check seems useless because it can never get false, but takes up quite some time..
    #    if oneIfTopicAssignmentsSupported(textCorpus, tLArr, zMat)!=1:
    #        return float("-inf")
    
    summand3 = 0.0
    for iteratingDoc in range(len(textCorpus)):
        summand3 += gammaln(len(activeTopics)/c_theta(len(activeTopics))*alphaTheta) - len(activeTopics)*gammaln(alphaTheta/c_theta(len(activeTopics)))
        
    summand4 = 0.0
    for iteratingDoc in range(len(textCorpus)):
        for iteratingTopic in activeTopics:
            summand4 += gammaln(alphaTheta/c_theta(len(activeTopics)) + \
                                numTopicOccurencesInDoc.get((iteratingDoc, iteratingTopic), 0.0))
            summand4 -= gammaln(len(activeTopics)/c_theta(len(activeTopics))*alphaTheta + \
                                numTopicOccurencesInDoc.get((iteratingDoc, iteratingTopic),0.0))
        
    summand5 = 0.0
    for iteratingTopic in activeTopics:
        summand5 += gammaln(numWordTypesActivatedInTopics[iteratingTopic]/c_f(numWordTypesActivatedInTopics[iteratingTopic])*alphaF)\
                             - numWordTypesActivatedInTopics[iteratingTopic] \
                                * gammaln(alphaF / c_f(numWordTypesActivatedInTopics[iteratingTopic]))

    summand6 = 0.0
    for iteratingTopic in activeTopics:
        for r in range(numWordTypesActivatedInTopics.get(iteratingTopic, 0)):
            topicsForWord = getNumTopicAssignmentsToWordType(
                                                    topic=iteratingTopic, 
                                                    wordType=getRthActiveWordTypeInTopic(
                                                                            r=r, 
                                                                            topic=iteratingTopic,
                                                                            zMat=zMat), 
                                                    tLArr=tLArr,
                                                    textCorpus=textCorpus)
            summand6 += gammaln(alphaF/c_f(numWordTypesActivatedInTopics[iteratingTopic]+topicsForWord))
            summand6 -= gammaln(numWordTypesActivatedInTopics[iteratingTopic]/c_f(numWordTypesActivatedInTopics[iteratingTopic]) + topicsForWord)
    
    return summand3 + summand4 + summand5 + summand6

