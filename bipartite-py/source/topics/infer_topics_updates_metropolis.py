'''
Created on Mar 31, 2014

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
from source.topics.infer_topics_updates import *

from state import *
from source.expressions import psiTildeFunction, kappaFunction, psiFunction
from numpy.ma.testutils import assert_almost_equal
from source.prob import flipCoin

PROPOSE_CREATE, PROPOSE_ADD, PROPOSE_DELETE = 1, 2, 3
proposalTypes = (PROPOSE_CREATE, PROPOSE_ADD, PROPOSE_DELETE)

def updateZs(textCorpus, samplingVariables, hyperParameters, limitUpdatesToWordTypes=None):
    
    """
    a Metropolis algorithm to update zMat's and tLArr's simultaneously 
    """
    samplingVariables.counts.assertConsistency(textCorpus, samplingVariables)

    print "corpus: ", textCorpus
    wordTypesOccuringInCorpus = textCorpus.getWordTypesOccuringInCorpus()
    for iteratingWordType in wordTypesOccuringInCorpus:
        if limitUpdatesToWordTypes is not None and iteratingWordType not in limitUpdatesToWordTypes:
            continue
        proposalTypeProportions = drawProposalTypeProportions(wordType=iteratingWordType, 
                                                          zMat=samplingVariables.zMat,
                                              activeTopics=samplingVariables.getActiveTopics())
        proposalType = drawProposalType(proposalTypeProportions)
        print "proposalTypeProportions:", proposalTypeProportions
        print "proposalType:", proposalType
        print "t assignments:", samplingVariables.tLArr
        print "active topics:", samplingVariables.getActiveTopics()
        print "zMat:", samplingVariables.zMat

        if proposalType == PROPOSE_CREATE: # draw new u
            proposeCreateAndAcceptOrReject(wordType=iteratingWordType, 
                                           textCorpus=textCorpus, 
                                           hyperParameters=hyperParameters,
                                           samplingVariables=samplingVariables,
                                           proposalTypeProportions=proposalTypeProportions, 
                                           numActiveTopicsForWordType=\
                                                samplingVariables.counts.numActiveTopicsForWordType)
        elif proposalType == PROPOSE_ADD: # draw u<1
            proposeAddAndAcceptOrReject(wordType=iteratingWordType, 
                                        textCorpus=textCorpus, 
                                        hyperParameters=hyperParameters,
                                        samplingVariables=samplingVariables,
                                        proposalTypeProportions=proposalTypeProportions, 
                                        numActiveTopicsForWordType=\
                                                samplingVariables.counts.numActiveTopicsForWordType)
        if proposalType == PROPOSE_DELETE: # set u=1
            proposeDeleteAndAcceptOrReject(wordType=iteratingWordType, 
                                        textCorpus=textCorpus, 
                                        hyperParameters=hyperParameters,
                                        samplingVariables=samplingVariables,
                                        proposalTypeProportions=proposalTypeProportions, 
                                        numActiveTopicsForWordType=\
                                                samplingVariables.counts.numActiveTopicsForWordType,
                                        numWordTypesActivatedInTopic=\
                                            samplingVariables.counts.numWordTypesActivatedInTopic)

def proposeAndAcceptOrReject(topic, isNewTopic, isDeletingTopic, wordType, textCorpus, 
                             hyperParameters, samplingVariablesWithRevertableChanges,
                             proposalTypeProportions, numActiveTopicsForWordType,
                             numWordTypesActivatedInTopic, logQiUToUTilde, logQiUTildeToU):
    samplingVariables = samplingVariablesWithRevertableChanges
    originalActiveTopics = samplingVariables.getActiveTopics()
    newActiveTopics = originalActiveTopics
    kiPlus = 0
    if isNewTopic:
        kiPlus = 1
        newActiveTopics = originalActiveTopics + [topic]
    if isDeletingTopic and numWordTypesActivatedInTopic[topic]<1.001:
        newActiveTopics = list(originalActiveTopics)
        newActiveTopics.remove(topic)
    samplingVariables.activateRevertableChanges(False)
    logProbWUGamma = computeLogProbWUGamma(kiPlus=0,
                                           wordType=wordType,
                                                textCorpus=textCorpus, 
                                                hyperParameters=hyperParameters,
                                                gammas=samplingVariables.gammas, 
                                                uMat=samplingVariables.uMat, 
                                                activeTopics=originalActiveTopics, 
                                                wArr=samplingVariables.wArr, 
                 numActiveTopicsForWordType=samplingVariables.counts.numActiveTopicsForWordType,
                 numWordTypesActivatedInTopic=samplingVariables.counts.numWordTypesActivatedInTopic)
    
    samplingVariables.activateRevertableChanges()
    logProbWUGammaTilde = computeLogProbWUGamma(kiPlus=kiPlus,
                                                wordType=wordType, 
                                                textCorpus=textCorpus, 
                                                uMat=samplingVariables.uMat, 
                                                activeTopics=originalActiveTopics, 
                                                gammas=samplingVariables.gammas, 
                                                wArr=samplingVariables.wArr, 
                                                hyperParameters=hyperParameters,
                 numActiveTopicsForWordType=samplingVariables.counts.numActiveTopicsForWordType,
                 numWordTypesActivatedInTopic=samplingVariables.counts.numWordTypesActivatedInTopic)
    
    # draw new topic assignment proposal: re-draw all occurrences of wordType
    LQi = samplingVariables.counts.docWordPosListForWordTypes[wordType]
    oldTopics = [samplingVariables.tLArr[docPos][wordPos] for (docPos,wordPos) in LQi]
    samplingVariables.activateRevertableChanges()
    logProbDrawingNewTopics = drawRevertableTopicProposalsAndUpdateCounts(LQi=LQi, 
                                                activeTopics=newActiveTopics,
                                                tLArr=samplingVariables.tLArr, 
                                                zMat=samplingVariables.zMat, 
                                                counts=samplingVariables.counts, 
                                                hyperParameters=hyperParameters, 
                                                textCorpus=textCorpus)
#    newTopics = [samplingVariables.tLArr[docPos][wordPos] for (docPos,wordPos) in LQi]
    samplingVariables.activateRevertableChanges(False)
    logProbRevertingTopics = computeLogProbOfDrawingTopics(LQi=LQi, 
                                                drawnTopics=oldTopics,
                                                activeTopics=originalActiveTopics,
                                                tLArr=samplingVariables.tLArr, 
                                                zMat=samplingVariables.zMat, 
                                                counts=samplingVariables.counts, 
                                                hyperParameters=hyperParameters, 
                                                textCorpus=textCorpus)
    samplingVariables.activateRevertableChanges()
    
    # compute logProbWoTTildeGivenZTilde and logProbWoTGivenZ
    logProbWoTTildeGivenZTilde = computeRelativeLogProbabilityForTWoGivenZ(
            activeTopics=newActiveTopics,
            textCorpus=textCorpus, 
            tLArr=samplingVariables.tLArr, 
            zMat=samplingVariables.zMat,
            gammas=samplingVariables.gammas, 
            wArr=samplingVariables.wArr, 
            alphaTheta=hyperParameters.alphaTheta, 
            alphaF=hyperParameters.alphaF,
            numWordTypesActivatedInTopics=samplingVariables.counts.numWordTypesActivatedInTopic,
            numTopicOccurencesInDoc=samplingVariables.counts.numTopicOccurencesInDoc,
            numTopicAssignmentsToWordType=samplingVariables.counts.numTopicAssignmentsToWordType,
            c_theta = expr.c_theta_K,
            c_f = expr.c_f_mj)
    samplingVariables.activateRevertableChanges(False)
    logProbWoTGivenZ = computeRelativeLogProbabilityForTWoGivenZ(
            activeTopics=originalActiveTopics,
            textCorpus=textCorpus, 
            tLArr=samplingVariables.tLArr, 
            zMat=samplingVariables.zMat,
            gammas=samplingVariables.gammas, 
            wArr=samplingVariables.wArr, 
            alphaTheta=hyperParameters.alphaTheta, 
            alphaF=hyperParameters.alphaF,
            numWordTypesActivatedInTopics=samplingVariables.counts.numWordTypesActivatedInTopic,
            numTopicOccurencesInDoc=samplingVariables.counts.numTopicOccurencesInDoc,
            numTopicAssignmentsToWordType=samplingVariables.counts.numTopicAssignmentsToWordType,
            c_theta = expr.c_theta_K,
            c_f = expr.c_f_mj)
   
    
    # compute probs of drawing topics t and t~
    ratio = math.exp(  logProbWUGammaTilde          - logProbWUGamma \
                     + logProbWoTTildeGivenZTilde   - logProbWoTGivenZ \
                     + logQiUTildeToU               - logQiUToUTilde\
                     + logProbRevertingTopics       - logProbDrawingNewTopics)
    if ratio >= 1 or flipCoin(ratio):
        print "accepted"
        samplingVariables.makePermanent()
        
        if isNewTopic:
            samplingVariables.createNewTopic(preparedPotentialTopicIndex=topic)
    
            # became unnecessary with the new u-based MH algo
#            for iteratingWordType in range(textCorpus.getVocabSize()):
#                samplingVariables.uMat[iteratingWordType, topic] = \
#                                prob.sampleFrom15(samplingVariables.gammas[:iteratingWordType+1], 
#                                                  samplingVariables.uMat[:iteratingWordType,topic],
#                                                  1,
#                                                  hyperParameters)
    else:
        print "rejected"
        samplingVariables.revert()

    
    # bugcheck:
    samplingVariables.counts.assertConsistency(textCorpus, samplingVariables)
    assert oneIfTopicAssignmentsSupported(textCorpus, samplingVariables.tLArr, 
                                          samplingVariables.zMat)==1
     
def proposeCreateAndAcceptOrReject(wordType, textCorpus, hyperParameters, samplingVariables,
                                   proposalTypeProportions, numActiveTopicsForWordType):

    newTopic = samplingVariables.preparePotentialNewTopic(activeForWord=wordType)

    newU = prob.sampleFrom15(gammas=[samplingVariables.gammas[wordType]], 
                             uxjList=[], 
                             mj=1, 
                             parameters=hyperParameters)
    oldU = 1 # any value will do
    gammaUSum = sum([samplingVariables.gammas[i] \
               for i in range(textCorpus.getVocabSize()) if i!=wordType])
    gammaUSum += samplingVariables.gammas[wordType] * newU 
    newW = np.random.gamma(1 - hyperParameters.sigma,
                           1.0/(hyperParameters.tau+gammaUSum)) 
    
    samplingVariables.revertableChangeInZ(wordType, newTopic, oldZ=0, newZ=1)
    samplingVariables.revertableChangeInU(wordType, newTopic, oldU=oldU, newU=newU)
    samplingVariables.revertableChangeInW(newTopic, oldW=1, newW=newW)

    # compute probs of moving from Z to ZTilde and vice versa
    logQiUToUTilde = math.log(proposalTypeProportions[PROPOSE_CREATE]\
                             *expr.lambdaFunction(newW, hyperParameters.alpha, 
                                                  hyperParameters.sigma, hyperParameters.tau)\
                             *newW\
                             *math.exp(-newW*gammaUSum))
    logQiUTildeToU = math.log(proposalTypeProportions[PROPOSE_DELETE] \
                             * 1.0/(numActiveTopicsForWordType[wordType]))

    proposeAndAcceptOrReject(topic=newTopic, 
                             isNewTopic=True, 
                             isDeletingTopic=False,
                             wordType=wordType,
                             textCorpus=textCorpus,
                             hyperParameters=hyperParameters, 
                             samplingVariablesWithRevertableChanges=samplingVariables,
                             proposalTypeProportions=proposalTypeProportions, 
                             numActiveTopicsForWordType=numActiveTopicsForWordType,
                             numWordTypesActivatedInTopic=\
                                            samplingVariables.counts.numWordTypesActivatedInTopic,
                             logQiUToUTilde=logQiUToUTilde, 
                             logQiUTildeToU=logQiUTildeToU)

        
def proposeAddAndAcceptOrReject(wordType, textCorpus, hyperParameters, samplingVariables,
                                   proposalTypeProportions, numActiveTopicsForWordType):
    addedTopic = drawTopicInactiveForWordType(wordType=wordType, 
                                              activeTopics=samplingVariables.getActiveTopics(), 
                                              zMat=samplingVariables.zMat, 
                                              numActiveTopicsForWordType=numActiveTopicsForWordType)
    newU = prob.sampleFrom15(gammas=samplingVariables.gammas, 
                             uxjList=samplingVariables.uMat[:wordType,addedTopic], 
                             mj=samplingVariables.counts.numWordTypesActivatedInTopic[addedTopic]+1, 
                             parameters=hyperParameters,
                             curWordType=wordType)
    gammaUSum = sum([samplingVariables.gammas[i]*samplingVariables.uMat[i,addedTopic] \
               for i in range(textCorpus.getVocabSize()) if i!=wordType])
    gammaUSum += samplingVariables.gammas[wordType] * newU 
    newW = np.random.gamma(1 - hyperParameters.sigma,
                           1.0/(hyperParameters.tau+gammaUSum)) 
    samplingVariables.revertableChangeInZ(wordType, addedTopic, oldZ=0, newZ=1)
    samplingVariables.revertableChangeInU(wordType, addedTopic, oldU=1, newU=newU)
    
    # compute probs of moving from Z to ZTilde and vice versa
    K = len(samplingVariables.getActiveTopics())
    Kinew = numActiveTopicsForWordType[wordType]
    gammaUSum = sum([samplingVariables.gammas[i]*samplingVariables.uMat[i,addedTopic] \
               for i in range(textCorpus.getVocabSize()) if i!=wordType])
    gammaUSum += samplingVariables.gammas[wordType]*newU
    print "K:", K
    print "Kinew:", Kinew
    samplingVariables.activateRevertableChanges()
    newMJ = samplingVariables.counts.numWordTypesActivatedInTopic[addedTopic]
    logQiUToUTilde = math.log(proposalTypeProportions[PROPOSE_ADD] / (K - (Kinew-1))\
                              *expr.lambdaFunction(newW, hyperParameters.alpha, 
                                                  hyperParameters.sigma, hyperParameters.tau)\
                             *(newW**newMJ)\
                             *math.exp(-newW*gammaUSum))
    reversalProposalTypeProportions = drawProposalTypeProportions(wordType, samplingVariables.zMat, samplingVariables.getActiveTopics())
    logQiUTildeToU = math.log(reversalProposalTypeProportions[PROPOSE_DELETE] / (Kinew))

    proposeAndAcceptOrReject(topic=addedTopic, 
                             isNewTopic=False, 
                             isDeletingTopic=False,
                             wordType=wordType,
                             textCorpus=textCorpus,
                             hyperParameters=hyperParameters, 
                             samplingVariablesWithRevertableChanges=samplingVariables,
                             proposalTypeProportions=proposalTypeProportions, 
                             numActiveTopicsForWordType=numActiveTopicsForWordType,
                             numWordTypesActivatedInTopic=\
                                            samplingVariables.counts.numWordTypesActivatedInTopic,
                             logQiUToUTilde=logQiUToUTilde, 
                             logQiUTildeToU=logQiUTildeToU)

def proposeDeleteAndAcceptOrReject(wordType, textCorpus, hyperParameters, samplingVariables,
                                   proposalTypeProportions, numActiveTopicsForWordType,
                                   numWordTypesActivatedInTopic):
    if numActiveTopicsForWordType[wordType] <= 1.0001:
        # there will be no valid deletion proposals
        return
    
    deletedTopic = drawTopicActiveForWordType(wordType=wordType, 
                                              activeTopics=samplingVariables.getActiveTopics(), 
                                              zMat=samplingVariables.zMat, 
                                              numActiveTopicsForWordType=numActiveTopicsForWordType)
    
    oldU = samplingVariables.uMat[wordType,deletedTopic]
    samplingVariables.revertableChangeInZ(wordType, deletedTopic, oldZ=1, newZ=0)
    samplingVariables.revertableChangeInU(wordType, deletedTopic, oldU=oldU, newU=1)
    samplingVariables.activateRevertableChanges(False)

    K = len(samplingVariables.getActiveTopics())
    Ki = numActiveTopicsForWordType[wordType]
    logQiUToUTilde = math.log(proposalTypeProportions[PROPOSE_DELETE] / K)
    samplingVariables.activateRevertableChanges()
    reversalProposalTypeProportions = drawProposalTypeProportions(wordType, samplingVariables.zMat, samplingVariables.getActiveTopics())
    curW = samplingVariables.wArr[deletedTopic]
    gammaUSum = sum([samplingVariables.gammas[i]*samplingVariables.uMat[i,deletedTopic] \
               for i in range(textCorpus.getVocabSize())])
    if numWordTypesActivatedInTopic[deletedTopic] <= 1: # if topic gets killed:
        logQiUTildeToU = math.log(reversalProposalTypeProportions[PROPOSE_CREATE]\
                                 *expr.lambdaFunction(curW, hyperParameters.alpha, 
                                                  hyperParameters.sigma, hyperParameters.tau)\
                                 *(curW**samplingVariables.counts.numWordTypesActivatedInTopic[deletedTopic])\
                                 *math.exp(-curW*gammaUSum))
    else: 
        logQiUTildeToU = math.log(reversalProposalTypeProportions[PROPOSE_ADD] / (K - Ki + 1)\
                                 *expr.lambdaFunction(curW, hyperParameters.alpha, 
                                                  hyperParameters.sigma, hyperParameters.tau)\
                                 *(curW**samplingVariables.counts.numWordTypesActivatedInTopic[deletedTopic])\
                                 *math.exp(-curW*gammaUSum))

    samplingVariables.activateRevertableChanges(True)
    proposeAndAcceptOrReject(topic=deletedTopic, 
                             isNewTopic=False, 
                             isDeletingTopic=True,
                             wordType=wordType,
                             textCorpus=textCorpus,
                             hyperParameters=hyperParameters, 
                             samplingVariablesWithRevertableChanges=samplingVariables,
                             proposalTypeProportions=proposalTypeProportions, 
                             numActiveTopicsForWordType=numActiveTopicsForWordType,
                             numWordTypesActivatedInTopic=\
                                            samplingVariables.counts.numWordTypesActivatedInTopic,
                             logQiUToUTilde=logQiUToUTilde, 
                             logQiUTildeToU=logQiUTildeToU)
    # remove dead topics
    samplingVariables.releaseDeadTopics()

def drawTopicActiveForWordType(wordType, activeTopics, zMat, numActiveTopicsForWordType):
    r = random.randint(1, numActiveTopicsForWordType[wordType])
    cnt = 0
    index = -1
    while cnt < r:
        index += 1
        if zMat[wordType, activeTopics[index]]>0.9:
            cnt += 1
    return activeTopics[index]
    
def drawTopicInactiveForWordType(wordType, activeTopics, zMat, numActiveTopicsForWordType):
    r = random.randint(1, len(activeTopics) - numActiveTopicsForWordType[wordType])
    cnt = 0
    index = -1
    while cnt < r:
        index += 1
        if zMat[wordType, activeTopics[index]]<0.1:
            cnt += 1
    return activeTopics[index]
    
def drawRevertableTopicProposalsAndUpdateCounts(LQi, activeTopics, tLArr, zMat, counts, 
                                                hyperParameters, textCorpus ):
    jointLogProb = 0.0
    for r in range(len(LQi)):
        iteratingDoc, iteratingWordPos = LQi[r]
        oldTopic=tLArr[iteratingDoc][iteratingWordPos]
        sampledTopic, logProb = sampleTGivenZT(
                    activeTopics=activeTopics,
                    doc=iteratingDoc, 
                    wordPos=iteratingWordPos,
                    alphaTheta=hyperParameters.alphaTheta, 
                    alphaF=hyperParameters.alphaF,
                    textCorpus=textCorpus,
                    tLArr=tLArr,
                    zMat=zMat,
                    excludeDocWordPositions=LQi[r:],
                    numWordTypesActivatedInTopics=counts.numWordTypesActivatedInTopic,
                    numTopicOccurencesInDoc=counts.numTopicOccurencesInDoc,
                    numTopicAssignmentsToWordType=counts.numTopicAssignmentsToWordType)
        jointLogProb += logProb
        tLArr[iteratingDoc].setRevertable(iteratingWordPos, sampledTopic)
        tLArr.activateRevertableChanges()
        
        counts.updateRevertableChangeInT(docPos=iteratingDoc, wordPos=iteratingWordPos, 
                                         wordType=textCorpus[iteratingDoc][iteratingWordPos], 
                                         oldTopic=oldTopic, newTopic=sampledTopic)
    
    return jointLogProb

def computeLogProbOfDrawingTopics(LQi, drawnTopics, activeTopics, tLArr, zMat, counts, 
                                                hyperParameters, textCorpus ):
    assert len(LQi) == len(drawnTopics)
    jointLogProb = 0.0
    for r in range(len(LQi)):
        iteratingDoc, iteratingWordPos = LQi[r]
        drawnTopic=drawnTopics[r]
        logProb = probDistributionTGivenZT(
                    activeTopics=activeTopics,
                    doc=iteratingDoc, 
                    wordPos=iteratingWordPos,
                    alphaTheta=hyperParameters.alphaTheta, 
                    alphaF=hyperParameters.alphaF,
                    textCorpus=textCorpus,
                    tLArr=tLArr,
                    zMat=zMat,
                    excludeDocWordPositions=LQi[r:],
                    numWordTypesActivatedInTopics=counts.numWordTypesActivatedInTopic,
                    numTopicOccurencesInDoc=counts.numTopicOccurencesInDoc,
                    numTopicAssignmentsToWordType=counts.numTopicAssignmentsToWordType)[activeTopics.index(drawnTopic)]
        jointLogProb += logProb
    return jointLogProb

def computeLogProbWUGamma(kiPlus, wordType, textCorpus, hyperParameters, gammas, wArr, uMat,  
                          activeTopics, numActiveTopicsForWordType, numWordTypesActivatedInTopic):
    logProbWUGamma = 0.0
    for iteratingWordType in range(textCorpus.getVocabSize()):
        if iteratingWordType==wordType:
            logProbWUGamma += (numActiveTopicsForWordType[wordType] + kiPlus)\
                                * math.log(gammas[iteratingWordType])
        else:
            logProbWUGamma += numActiveTopicsForWordType[wordType]\
                                * math.log(gammas[iteratingWordType])
    for iteratingTopic in activeTopics:
        logProbWUGamma += numWordTypesActivatedInTopic[iteratingTopic]\
                            * math.log(wArr[iteratingTopic])
        logProbWUGamma += math.log(expr.lambdaFunction(wArr[iteratingTopic], 
                                              hyperParameters.alpha, 
                                              hyperParameters.sigma, 
                                              hyperParameters.tau) )      
        for iteratingWordType in range(textCorpus.getVocabSize()):
            logProbWUGamma -= wArr[iteratingTopic]\
                                * gammas[iteratingWordType]\
                                * uMat[iteratingWordType,iteratingTopic]
    return logProbWUGamma

            
#def computeLogProbZGivenWGamma(wordType, kiPlus, zMat, activeTopics, gammas, wArr, alpha,sigma,tau):
#    logProbZGivenWGamma = 0.0
#    for iteratingTopic in activeTopics:
#        logProbZGivenWGamma += (1.0 - zMat[wordType, iteratingTopic]) \
#                                * (-gammas[wordType] * wArr[iteratingTopic])
#        logProbZGivenWGamma += zMat[wordType, iteratingTopic] \
#                            * math.log(1.0 - math.exp(-gammas[wordType] * wArr[iteratingTopic]))
#        logProbZGivenWGamma += logPoissonPsiFunc(kiPlus=kiPlus, 
#                                                 wordType=wordType, 
#                                                 gammas=gammas, 
#                                                 alpha=alpha, 
#                                                 sigma=sigma, 
#                                                 tau=tau)
#    return logProbZGivenWGamma
    
def logPoissonPsiFunc(kiPlus, wordType, gammas, alpha, sigma, tau):
        lamPoisson = expr.psiTildeFunction(t=gammas[wordType], 
                                    b=sum(gammas)-gammas[wordType],
                                    alpha=alpha,
                                    sigma=sigma,
                                    tau=tau)
        return math.log(lamPoisson**kiPlus * math.exp(-lamPoisson) / math.factorial(kiPlus))
  
    
def drawProposalTypeProportions(wordType, zMat, activeTopics):
    print "getNumActiveTopicsForWordType(wordType, zMat, activeTopics):", getNumActiveTopicsForWordType(wordType, zMat, activeTopics)
    print "len(activeTopics)", len(activeTopics)
    if getNumActiveTopicsForWordType(wordType, zMat, activeTopics) < len(activeTopics):
        return {PROPOSE_CREATE: 1.0/3, 
                PROPOSE_ADD:    1.0/3, 
                PROPOSE_DELETE: 1.0/3}
    else:
        return {PROPOSE_CREATE: 0.5, 
                PROPOSE_ADD:    0.0, 
                PROPOSE_DELETE: 0.5}
        
def drawProposalType(probs):
    keys = list(probs.keys())
    vals = list([probs[key] for key in keys])
    return keys[int(np.nonzero(np.random.multinomial(1, vals))[0][0])]


def computeRelativeLogProbabilityForTWoGivenZ(activeTopics, textCorpus, tLArr, zMat, gammas, 
                                       wArr, alphaTheta, alphaF, numWordTypesActivatedInTopics,
                                       numTopicOccurencesInDoc, numTopicAssignmentsToWordType,
                                       c_theta, c_f):
#    # this check seems useless because it can never get false, but takes up quite some time..
    if oneIfTopicAssignmentsSupported(textCorpus, tLArr, zMat)!=1:
        return float("-inf")

    K = len(activeTopics)
    c_theta_K = c_theta(K)
    
    summand1 = 0.0
    summand2 = 0.0
    for iteratingDocPos in range(len(textCorpus)):
        summand1 += gammaln(K/c_theta_K*alphaTheta)
        summand1 -= K * gammaln(alphaTheta / c_theta_K)
        
        for iteratingTopic in activeTopics:
            summand2 += gammaln(alphaTheta/c_theta_K + \
                                numTopicOccurencesInDoc[iteratingDocPos, iteratingTopic])
            summand2 -= gammaln(K/c_theta_K*alphaTheta + \
                                numTopicOccurencesInDoc[iteratingDocPos, iteratingTopic])
    summand3 = 0.0
    summand4 = 0.0
    summand5 = 0.0
    for iteratingTopic in activeTopics:
        mj = numWordTypesActivatedInTopics[iteratingTopic]
        summand3 += gammaln(mj / c_f(mj) * alphaF)
        summand3 -= mj * gammaln(alphaF / c_f(mj))
        
        for r in range(mj):
            numWordsTypeTopicCoOccurences = numTopicAssignmentsToWordType[
                    (getRthActiveWordTypeInTopic(r=r, topic=iteratingTopic, zMat=zMat),
                     iteratingTopic)]
                
            summand4 += gammaln(alphaF/c_f(mj) + numWordsTypeTopicCoOccurences)
        numTopicOccurences = 0
        for iteratingDocPos in range(len(textCorpus)):
            numTopicOccurences += numTopicOccurencesInDoc[iteratingDocPos]
        summand5 -= gammaln(mj / c_f(mj) + numTopicOccurences)
    return summand1 + summand2 + summand3 + summand4 + summand5
