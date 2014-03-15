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

from infer_topics_state import *
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

def updateZs(textCorpus, samplingVariables, hyperParameters):
    """
    a Metropolis algorithm to update zMat's and tLArr's simultaneously 
    """
    for iteratingWordType in range(textCorpus.getVocabSize()):
        for iteratingTopic in samplingVariables.getActiveTopics():
            # skip the case where only topic j is active for word i: we need at
            # least one topic in which each word is activated
            if isOnlyActivatedTopicForWordType(wordType=iteratingWordType,
                                                topic=iteratingTopic, 
                                                samplingVariables=samplingVariables):
                continue

            # skip this topic if only one topic is active: there will be no valid proposals
            if len(samplingVariables.getActiveTopics()) == 1:
                break
            
            # core events take place here:
            samplingVariables.releaseDeadTopics()
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

def isOnlyActivatedTopicForWordType(wordType, topic, samplingVariables):
    return getNumActiveTopicsForWordType(wordType, 
                                 samplingVariables.zMat, 
                                 samplingVariables.getActiveTopics()) == 1 \
        and abs(samplingVariables.zMat[wordType,topic] - 1) <= 1e-6
        
def proposeAndAcceptOrReject(iteratingWordType, iteratingTopic, samplingVariables, 
                             textCorpus, hyperParameters):
    LQij = samplingVariables.counts.docWordPosListForTopicAssignments[iteratingWordType, iteratingTopic]
#    for iteratingDoc in range(len(textCorpus)):
#        for iteratingWordPos in range(len(textCorpus[iteratingDoc])):
#            if textCorpus[iteratingDoc][iteratingWordPos]==iteratingWordType \
#                    and samplingVariables.tLArr[iteratingDoc][iteratingWordPos]==iteratingTopic:
#                LQij.append((iteratingDoc, iteratingWordPos))

    # switch z_ij between 0 and 1
    zTilde_ij = 1 - samplingVariables.zMat[iteratingWordType, iteratingTopic]
#    zTilde = samplingVariables.zMat.copy()
#    zTilde[iteratingWordType, iteratingTopic] = 1 - zTilde[iteratingWordType, iteratingTopic]
#    tTilde = copy.deepcopy(samplingVariables.tLArr)
    
    # resample topics
    # careful: these are changed in-place (don't want to re-allocate a new array every
    # single step), so must be changed back immediately after
    numWordTypesActivatedInTopics = samplingVariables.counts.numWordTypesActivatedInTopic
#    if zTilde[iteratingWordType, iteratingTopic]==1:
    if zTilde_ij==1:
        numWordTypesActivatedInTopics.setRevertable(iteratingTopic,
                                                    numWordTypesActivatedInTopics[iteratingTopic]+1)
    else: 
        numWordTypesActivatedInTopics.setRevertable(iteratingTopic,
                                                    numWordTypesActivatedInTopics[iteratingTopic]-1)
    numWordTypesActivatedInTopics.activateRevertableChanges()
    
    samplingVariables.zMat[iteratingWordType, iteratingTopic] = zTilde_ij
    for r in range(len(LQij)):
        iteratingDoc, iteratingWordPos = LQij[r]
        samplingVariables.tLArr[iteratingDoc].setRevertable(iteratingWordPos, sampleTGivenZT(
                    activeTopics=samplingVariables.getActiveTopics(),
                    doc=iteratingDoc, 
                    wordPos=iteratingWordPos,
                    alphaTheta=hyperParameters.alphaTheta, 
                    alphaF=hyperParameters.alphaF,
                    textCorpus=textCorpus,
                    tLArr=samplingVariables.tLArr,
                    zMat=samplingVariables.zMat,
                    excludeDocWordPositions=LQij[r+1:],
                    numWordTypesActivatedInTopics=numWordTypesActivatedInTopics,
                    numTopicOccurencesInDoc=samplingVariables.counts.numTopicOccurencesInDoc,
                    numTopicAssignmentsToWordType=\
                                samplingVariables.counts.numTopicAssignmentsToWordType))
        samplingVariables.tLArr.activateRevertableChanges()
        # update counts
        samplingVariables.counts.numTopicAssignmentsToWordType.addRevertable(
                (textCorpus[iteratingDoc][iteratingWordPos],
                 samplingVariables.tLArr[iteratingDoc][iteratingWordPos]),
                +1)
        samplingVariables.tLArr.activateRevertableChanges(False)
        samplingVariables.counts.numTopicAssignmentsToWordType.addRevertable(
                (textCorpus[iteratingDoc][iteratingWordPos],
                 samplingVariables.tLArr[iteratingDoc][iteratingWordPos]),
                -1)

        samplingVariables.tLArr.activateRevertableChanges()
        samplingVariables.counts.numTopicOccurencesInDoc.addRevertable((iteratingDoc,
                                samplingVariables.tLArr[iteratingDoc][iteratingWordPos]), + 1)
        samplingVariables.tLArr.activateRevertableChanges(False)
        samplingVariables.counts.numTopicOccurencesInDoc.addRevertable((iteratingDoc,
                                samplingVariables.tLArr[iteratingDoc][iteratingWordPos]), - 1)
    samplingVariables.tLArr.activateRevertableChanges()
    samplingVariables.counts.numTopicOccurencesInDoc.activateRevertableChanges()
    samplingVariables.counts.numTopicAssignmentsToWordType.activateRevertableChanges()
    
    # compute relative probabilities
    activeTopicsTilde = list(samplingVariables.getActiveTopics())
    topicCouldBeDying = (numWordTypesActivatedInTopics[iteratingTopic] == 0)
    if topicCouldBeDying:
        activeTopicsTilde.remove(iteratingTopic)
        logprob1 = computeRelativeLogProbabilityForTZSafe(
                        activeTopics=activeTopicsTilde,
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
                        numTopicOccurencesInDoc=samplingVariables.counts.numTopicOccurencesInDoc,
                        numTopicAssignmentsToWordType=samplingVariables.counts.numTopicAssignmentsToWordType)
    else:
        logprob1 = computeRelativeLogProbabilityForTZSimplified(
                        activeTopics=activeTopicsTilde,
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
                        numTopicOccurencesInDoc=samplingVariables.counts.numTopicOccurencesInDoc,
                        numTopicAssignmentsToWordType=samplingVariables.counts.numTopicAssignmentsToWordType)

    # change back to the original state
    samplingVariables.tLArr.activateRevertableChanges(False)
    numWordTypesActivatedInTopics.activateRevertableChanges(False)
    samplingVariables.counts.numTopicOccurencesInDoc.activateRevertableChanges(False)
    samplingVariables.counts.numTopicAssignmentsToWordType.activateRevertableChanges(False)
    samplingVariables.zMat[iteratingWordType, iteratingTopic] = 1-zTilde_ij
    if topicCouldBeDying:
        logprob2 = computeRelativeLogProbabilityForTZSafe(
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
                    numTopicOccurencesInDoc=samplingVariables.counts.numTopicOccurencesInDoc,
                    numTopicAssignmentsToWordType=samplingVariables.counts.numTopicAssignmentsToWordType)
    else:
        logprob2 = computeRelativeLogProbabilityForTZSimplified(
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
                    numTopicOccurencesInDoc=samplingVariables.counts.numTopicOccurencesInDoc,
                    numTopicAssignmentsToWordType=samplingVariables.counts.numTopicAssignmentsToWordType)
    if logprob1 > logprob2: 
        ratio=1.0
    elif logprob1 == float("-inf"):
        ratio = 0.0
    else:
        ratio = math.exp(logprob1 - logprob2)
    # accept or reject
    if prob.flipCoin(ratio):
        # accept
        samplingVariables.zMat[iteratingWordType, iteratingTopic] = zTilde_ij
        samplingVariables.tLArr.makePermanent()
        samplingVariables.counts.docWordPosListForTopicAssignments[iteratingWordType, iteratingTopic] = []
        for (doc,wordPos) in LQij:
            newTopic = samplingVariables.tLArr[doc][wordPos]
            if (iteratingWordType, newTopic) not in samplingVariables.counts.docWordPosListForTopicAssignments:
                samplingVariables.counts.docWordPosListForTopicAssignments[iteratingWordType, newTopic] = []
            samplingVariables.counts.docWordPosListForTopicAssignments[iteratingWordType, newTopic].append((doc, wordPos))
        numWordTypesActivatedInTopics.makePermanent()
        samplingVariables.counts.numTopicOccurencesInDoc.makePermanent()
        samplingVariables.counts.numTopicAssignmentsToWordType.makePermanent()
        if samplingVariables.zMat[iteratingWordType, iteratingTopic]==1:
            samplingVariables.counts.numActiveTopicsForWordType[iteratingWordType] += 1
        else:
            samplingVariables.counts.numActiveTopicsForWordType[iteratingWordType] -= 1
    else:
        # reject
        # revert changes to samplingVariables.counts.numTopicAssignmentsToWordType:
        samplingVariables.tLArr.revert()
        numWordTypesActivatedInTopics.revert()
        samplingVariables.counts.numTopicOccurencesInDoc.revert()
        samplingVariables.counts.numTopicAssignmentsToWordType.revert()

def createNewTopics(iteratingWordType, textCorpus, samplingVariables, hyperParameters):
    numNewTopics = sampleTruncatedNumNewTopicsLog(
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
#    print "nr new topics", numNewTopics
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

def updateTs(textCorpus, samplingVariables, hyperParameters):
    for iteratingDocument in range(len(textCorpus)):
        for iteratingWordPos in range(len(textCorpus[iteratingDocument])):
            currentWordType = textCorpus[iteratingDocument][iteratingWordPos]
            prevTopic = samplingVariables.tLArr[iteratingDocument][iteratingWordPos]
            newTopic = \
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
            samplingVariables.counts.docWordPosListForTopicAssignments[currentWordType,prevTopic].remove((iteratingDocument,iteratingWordPos))
            if (currentWordType,newTopic) not in samplingVariables.counts.docWordPosListForTopicAssignments:
                samplingVariables.counts.docWordPosListForTopicAssignments[currentWordType,newTopic] = []
            samplingVariables.counts.docWordPosListForTopicAssignments[currentWordType,newTopic].append((iteratingDocument,iteratingWordPos))
            samplingVariables.counts.numTopicAssignmentsToWordType[ \
                                    currentWordType,
                                    prevTopic] \
                            = samplingVariables.counts.numTopicAssignmentsToWordType.get(( \
                                    currentWordType,
                                    prevTopic),0) - 1
            samplingVariables.counts.numTopicAssignmentsToWordType[ \
                                currentWordType,
                                newTopic] \
                        = samplingVariables.counts.numTopicAssignmentsToWordType.get(( \
                                currentWordType,
                                newTopic), 0) + 1
            samplingVariables.counts.numTopicOccurencesInDoc[iteratingDocument,prevTopic] = \
                    samplingVariables.counts.numTopicOccurencesInDoc.get((iteratingDocument,prevTopic),0) \
                    -1
            samplingVariables.counts.numTopicOccurencesInDoc[iteratingDocument,newTopic] = \
                    samplingVariables.counts.numTopicOccurencesInDoc.get((iteratingDocument,newTopic),0) \
                    +1

    
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
    
def sampleTGivenZT(activeTopics, doc, wordPos, alphaTheta, alphaF, textCorpus, tLArr, 
                   zMat, numWordTypesActivatedInTopics, numTopicAssignmentsToWordType,
                   numTopicOccurencesInDoc,
                   c_theta = expr.c_theta_K,
                   c_f = expr.c_f_mj,
                   excludeDocWordPositions=[]):
    unnormalizedTopicProbs1, unnormalizedTopicProbs2 = [], []
    wordType = textCorpus[doc][wordPos]
    for iteratingTopic in activeTopics:
        if abs(zMat[wordType,iteratingTopic] - 0) <= 1e-6:
            unnormalizedTopicProbs1.append(0.0)
            unnormalizedTopicProbs2.append(0.0)
        else:
            numTopicAssignmentsToDocCount = numTopicOccurencesInDoc.get((doc, iteratingTopic), 0)
            numTopicAssignmentsToWordTypeCount = GibbsCounts.getNumTopicAssignmentsToWordTypeExcl(\
                                        wordType=wordType, topic=iteratingTopic, tLArr=tLArr,
                                        textCorpus=textCorpus, 
                                        numTopicAssignmentsToWordTypeDict=numTopicAssignmentsToWordType,
                                        excludeDocWordPositions=[(doc,wordPos)] + excludeDocWordPositions)
            summand1 = math.log(alphaTheta/c_theta(len(activeTopics)) + numTopicAssignmentsToDocCount)
            summand2 = gammaln(alphaF/c_f(numWordTypesActivatedInTopics[iteratingTopic]) + numTopicAssignmentsToWordTypeCount + 1.0)
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
    return activeTopics[np.nonzero(np.random.multinomial(1, normalizedTopicProbs2))[0][0]]

def computeRelativeLogProbabilityForTZSafe(activeTopics, textCorpus, wordType, topic, tLArr, zMat, 
                                       gammas, wArr, alphaTheta, alphaF, 
                                       numWordTypesActivatedInTopics, numTopicOccurencesInDoc,
                                       numTopicAssignmentsToWordType,
                                       c_theta = expr.c_theta_K,
                                       c_f = expr.c_f_mj ):
    
    # this check seems useless because it can never get false, but takes up quite some time..
    if oneIfTopicAssignmentsSupported(textCorpus, tLArr, zMat)!=1:
        return float("-inf")
    
    summand1 = zMat[wordType,topic] * math.log(1.0 - math.exp(-gammas[wordType]*wArr[topic]))
    
    summand2 = -(1-zMat[wordType,topic])*gammas[wordType]*wArr[topic]
    
    summand3 = 0.0
    for iteratingDoc in range(len(textCorpus)):
        summand3 += gammaln(len(activeTopics)/c_theta(len(activeTopics)) * alphaTheta)
    
    summand4 = -len(activeTopics) * gammaln(alphaTheta/c_theta(len(activeTopics)))
     
    summand5 = 0.0 
    for iteratingDoc in range(len(textCorpus)):
        for iteratingTopic in activeTopics:
            summand5 += gammaln(alphaTheta/c_theta(len(activeTopics)) + \
                                numTopicOccurencesInDoc.get((iteratingDoc, iteratingTopic), 0.0))
            summand5 -= gammaln(len(activeTopics)/c_theta(len(activeTopics))*alphaTheta + \
                                numTopicOccurencesInDoc.get((iteratingDoc, iteratingTopic),0.0))
    
    if numWordTypesActivatedInTopics[topic] > 0:
        summand6 = gammaln(numWordTypesActivatedInTopics[topic] \
                           /  c_f(numWordTypesActivatedInTopics[topic])) * alphaF
        summand7 = -numWordTypesActivatedInTopics[topic] \
                        * gammaln(alphaF / c_f(numWordTypesActivatedInTopics[topic]))
    else:
        summand6, summand7 = 0, 0
        
    summand8 = 0.0
    for iteratingTopic in activeTopics:
        numWordsAssignedToTopic = 0.0
        for r in range(numWordTypesActivatedInTopics[iteratingTopic]):
            numWordsAssignedToTopic = +numTopicAssignmentsToWordType[(iteratingTopic, 
                                                               getRthActiveWordTypeInTopic(
                                                                            r=r, 
                                                                            topic=iteratingTopic,
                                                                            zMat=zMat))]

        for r in range(numWordTypesActivatedInTopics[iteratingTopic]):
            topicsForWord = numTopicAssignmentsToWordType[(iteratingTopic, 
                                                               getRthActiveWordTypeInTopic(
                                                                            r=r, 
                                                                            topic=iteratingTopic,
                                                                            zMat=zMat))]
            summand8 += gammaln(alphaF/c_f(numWordTypesActivatedInTopics[iteratingTopic])+topicsForWord)
            
        summand8 += -gammaln(numWordTypesActivatedInTopics[iteratingTopic] \
                            /c_f(numWordTypesActivatedInTopics[iteratingTopic]) + numWordsAssignedToTopic)
    
    return summand1 + summand2 + summand3 + summand4 + summand5 + summand6 + summand7 + summand8



def computeRelativeLogProbabilityForTZSimplified(activeTopics, textCorpus, wordType, topic, tLArr, zMat, 
                                       gammas, wArr, alphaTheta, alphaF, 
                                       numWordTypesActivatedInTopics, numTopicOccurencesInDoc,
                                       numTopicAssignmentsToWordType,
                                       c_theta = expr.c_theta_K,
                                       c_f = expr.c_f_mj ):
    # use only if no topic dies:
    
    # this check seems useless because it can never get false, but takes up quite some time..
    if oneIfTopicAssignmentsSupported(textCorpus, tLArr, zMat)!=1:
        return float("-inf")
    
    summand1 = zMat[wordType,topic] * math.log(1.0 - math.exp(-gammas[wordType]*wArr[topic]))
    
    summand2 = -(1-zMat[wordType,topic])*gammas[wordType]*wArr[topic]
    
    summand3 = 0.0 
    for iteratingDoc in range(len(textCorpus)):
        for iteratingTopic in activeTopics:
            summand3 += gammaln(alphaTheta/c_theta(len(activeTopics)) + \
                                numTopicOccurencesInDoc.get((iteratingDoc, iteratingTopic), 0.0))
            summand3 -= gammaln(len(activeTopics)/c_theta(len(activeTopics))*alphaTheta + \
                                numTopicOccurencesInDoc.get((iteratingDoc, iteratingTopic),0.0))
    
    summand4 = gammaln(numWordTypesActivatedInTopics[topic] \
                       /  c_f(numWordTypesActivatedInTopics[topic])) * alphaF

    summand5 = -numWordTypesActivatedInTopics[topic] \
                    * gammaln(alphaF / c_f(numWordTypesActivatedInTopics[topic]))
        
    summand6 = 0.0
    for iteratingTopic in activeTopics:
        numWordsAssignedToTopic = 0.0
        for r in range(numWordTypesActivatedInTopics[iteratingTopic]):
            numWordsAssignedToTopic = +numTopicAssignmentsToWordType[(iteratingTopic, 
                                                               getRthActiveWordTypeInTopic(
                                                                            r=r, 
                                                                            topic=iteratingTopic,
                                                                            zMat=zMat))]

        for r in range(numWordTypesActivatedInTopics[iteratingTopic]):
            topicsForWord = numTopicAssignmentsToWordType[(iteratingTopic, 
                                                               getRthActiveWordTypeInTopic(
                                                                            r=r, 
                                                                            topic=iteratingTopic,
                                                                            zMat=zMat))]
            summand6 += gammaln(alphaF/c_f(numWordTypesActivatedInTopics[iteratingTopic])+topicsForWord)
            
    summand7 = -gammaln(numWordTypesActivatedInTopics[iteratingTopic] \
                        /c_f(numWordTypesActivatedInTopics[iteratingTopic]) + numWordsAssignedToTopic)
    
    return summand1 + summand2 + summand3 + summand4 + summand5 + summand6 + summand7


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


def sampleTruncatedNumNewTopicsLog(activeTopics, textCorpus, tLArr, alphaTheta, wordType,
                                gammas, alpha, sigma, tau, numTopicOccurencesInDoc,
                                c_theta=expr.c_theta_K, cutoff=30):
    
    logProbs = []
    k = len(activeTopics)
    for kiPlus in range(cutoff):
        kPlusKPlus = len(activeTopics) + kiPlus
        
        mainSummand = 0.0
        
        for iteratingDoc in range(len(textCorpus)):
            mainSummand += -k * gammaln(alphaTheta/c_theta(kPlusKPlus))
            
            mainSummand += gammaln(kPlusKPlus/c_theta(kPlusKPlus)*alphaTheta)
            
            innerSum = 0.0
            for j in activeTopics:
                innerSum += numTopicOccurencesInDoc.get((iteratingDoc, j),0)

            mainSummand += -gammaln(innerSum + kPlusKPlus/c_theta(kPlusKPlus)*alphaTheta)
            
            for topic in activeTopics:
                mainSummand += gammaln(alphaTheta/c_theta(kPlusKPlus) + numTopicOccurencesInDoc.get((iteratingDoc, topic),0))
        lamPoisson = expr.psiTildeFunction(t=gammas[wordType], 
                                    b=sum(gammas)-gammas[wordType],
                                    alpha=alpha,
                                    sigma=sigma,
                                    tau=tau)
        mainFactor2 = lamPoisson**kiPlus * math.exp(-lamPoisson) / math.factorial(kiPlus)
        logProbs.append(mainSummand + math.log(mainFactor2))
    maxLog = max(logProbs)
    for i in range(len(logProbs)): logProbs[i] -= maxLog
    unnormalizedProbs = [math.exp(v) for v in logProbs]
    normalizer = sum(unnormalizedProbs)
    normalizedProbs = [p / normalizer for p in unnormalizedProbs]
    return np.nonzero(np.random.multinomial(1, normalizedProbs))[0][0]

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

