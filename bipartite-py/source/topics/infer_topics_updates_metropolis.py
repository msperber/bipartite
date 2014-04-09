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
from source.topics.generate_topics import sampleNextU

PROPOSE_CREATE, PROPOSE_ADD, PROPOSE_DELETE = 1, 2, 3

def updateZsCorrected(textCorpus, samplingVariables, hyperParameters, limitUpdatesToWordTypes=None):
    
    """
    a Metropolis algorithm to update zMat's and tLArr's simultaneously 
    """
    for iteratingWordType in range(textCorpus.getVocabSize()):
        if limitUpdatesToWordTypes is not None and iteratingWordType not in limitUpdatesToWordTypes:
            continue
        
        proposalTypeProportions = drawProposalTypeProportions(wordType=iteratingWordType, 
                                                          zMat=samplingVariables.zMat,
                                                          samplingVariables.getActiveTopics())
        proposalType = drawProposalType(proposalTypeProportions)
        
        if proposalType == PROPOSE_CREATE:
            proposeCreateAndAcceptOrReject()
        elif proposalType == PROPOSE_ADD:
            proposeAddAndAcceptOrReject()
        if proposalType == PROPOSE_DELETE:
            proposeDeleteAndAcceptOrReject()
        
            # remove dead topics
            samplingVariables.releaseDeadTopics()
 
def proposeAndAcceptOrReject(topic, isNewTopic, wordType, textCorpus, hyperParameters, 
                             samplingVariablesWithRevertableChanges,
                             proposalTypeProportions, numActiveTopicsForWordType,
                             logQZToZTilde, logQZTildeToZ):
    samplingVariables = samplingVariablesWithRevertableChanges
    originalActiveTopics = samplingVariables.getActiveTopics()
    newActiveTopics = originalActiveTopics
    kiPlus = 0
    if isNewTopic:
        kiPlus = 1
        newActiveTopics = originalActiveTopics + [topic]
    samplingVariables.activateRevertableChanges(False)
    logProbZGivenWGamma = computeLogProbZGivenWGamma(wordType=wordType, 
                                                     kiPlus=0,
                                                     zMat=samplingVariables.zMat, 
                                                     activeTopics=originalActiveTopics, 
                                                     gammas=samplingVariables.gammas, 
                                                     wArr=samplingVariables.wArr, 
                                                     alpha=hyperParameters.alpha, 
                                                     sigma=hyperParameters.sigma, 
                                                     tau=hyperParameters.tau)
    samplingVariables.activateRevertableChanges()
    logProbZTildeGivenWGamma = computeLogProbZGivenWGamma(wordType=wordType, 
                                                     kiPlus=kiPlus,
                                                     zMat=samplingVariables.zMat, 
                                                     activeTopics=originalActiveTopics, 
                                                     gammas=samplingVariables.gammas, 
                                                     wArr=samplingVariables.wArr, 
                                                     alpha=hyperParameters.alpha, 
                                                     sigma=hyperParameters.sigma, 
                                                     tau=hyperParameters.tau)
    
    # draw new topic assignment proposal: re-draw all occurrences of wordType
    LQi = samplingVariables.counts.docWordPosListForWordTypes[wordType]
    oldTopics = [samplingVariables.tLArr[docPos][wordPos] for (docPos,wordPos) in LQi]
    samplingVariables.activateRevertableChanges()
    logProbDrawingNewTopics = drawRevertableTopicProposalsAndUpdateCounts(LQi=LQi, 
                                                samplingVariables=samplingVariables, 
                                                hyperParameters=hyperParameters, 
                                                textCorpus=textCorpus,
                                                numWordTypesActivatedInTopics=\
                                            samplingVariables.counts.numWordTypesActivatedInTopic)
#    newTopics = [samplingVariables.tLArr[docPos][wordPos] for (docPos,wordPos) in LQi]
    samplingVariables.activateRevertableChanges(False)
    logProbRevertingTopics = computeLogProbOfDrawingTopics(Qi=LQi, 
                                                drawnTopics=oldTopics,
                                                samplingVariables=samplingVariables, 
                                                hyperParameters=hyperParameters, 
                                                textCorpus=textCorpus,
                                                numWordTypesActivatedInTopics=\
                                            samplingVariables.counts.numWordTypesActivatedInTopic)
    samplingVariables.activateRevertableChanges()
    
    # compute logProbWoTTildeGivenZTilde and logProbWoTGivenZ
    logProbWoTTildeGivenZTilde = computeRelativeLogProbabilityForTZ(
            activeTopics=newActiveTopics,
            textCorpus=textCorpus, 
            tLArr=samplingVariables.tLArr, 
            zMat=samplingVariables.zMat,
            gammas=samplingVariables.gammas, 
            wArr=samplingVariables.wArr, 
            alphaTheta=hyperParameters.alphaTheta, 
            alphaF=hyperParameters.alphaF,
            numWordTypesActivatedInTopics=samplingVariables.counts.numWordTypesActivatedInTopics,
            numTopicOccurencesInDoc=samplingVariables.counts.numTopicOccurencesInDoc,
            numTopicAssignmentsToWordType=samplingVariables.counts.numTopicAssignmentsToWordType)
    samplingVariables.activateRevertableChanges(False)
    logProbWoTGivenZ = computeRelativeLogProbabilityForTZ(
            activeTopics=originalActiveTopics,
            textCorpus=textCorpus, 
            tLArr=samplingVariables.tLArr, 
            zMat=samplingVariables.zMat,
            gammas=samplingVariables.gammas, 
            wArr=samplingVariables.wArr, 
            alphaTheta=hyperParameters.alphaTheta, 
            alphaF=hyperParameters.alphaF,
            numWordTypesActivatedInTopics=samplingVariables.counts.numWordTypesActivatedInTopics,
            numTopicOccurencesInDoc=samplingVariables.counts.numTopicOccurencesInDoc,
            numTopicAssignmentsToWordType=samplingVariables.counts.numTopicAssignmentsToWordType)
   
    
    # compute probs of drawing topics t and t~
    ratio = math.exp(  logProbZTildeGivenWGamma   - logProbZGivenWGamma \
                     + logProbWoTTildeGivenZTilde - logProbWoTGivenZ \
                     + logQZTildeToZ              - logQZToZTilde\
                     + logProbRevertingTopics     - logProbDrawingNewTopics)
    if ratio >= 1 or flipCoin(ratio):
        samplingVariables.makePermanent()
        
        if isNewTopic:
            samplingVariables.createNewTopic(preparedPotentialTopicIndex=topic)
    
            for iteratingWordType in range(len(textCorpus.getVocabSize())):
                samplingVariables.uMat[iteratingWordType, topic] = \
                                prob.sampleFrom15(samplingVariables.gammas[:iteratingWordType+1], 
                                                  samplingVariables.uMat[:iteratingWordType,topic],
                                                  1,
                                                  hyperParameters)
    else:
        samplingVariables.revert()
        
def proposeCreateAndAcceptOrReject(wordType, textCorpus, hyperParameters,samplingVariables,
                                   proposalTypeProportions, numActiveTopicsForWordType):

    newTopic = samplingVariables.preparePotentialNewTopic(activeForWord=wordType)
    samplingVariables.counts.updateRevertableChangeInZ(wordType, newTopic, 0, 1)

    # compute probs of moving from Z to ZTilde and vice versa
    logQZToZTilde = math.log(proposalTypeProportions[PROPOSE_CREATE])
    logQZTildeToZ = math.log(proposalTypeProportions[PROPOSE_DELETE] \
                             * 1.0/(1.0+numActiveTopicsForWordType[wordType]))

    proposeAndAcceptOrReject(topic=newTopic, 
                             isNewTopic=True, 
                             wordType=wordType,
                             textCorpus=textCorpus,
                             hyperParameters=hyperParameters, 
                             samplingVariablesWithRevertableChanges=samplingVariables,
                             proposalTypeProportions=proposalTypeProportions, 
                             numActiveTopicsForWordType=numActiveTopicsForWordType,
                             logQZToZTilde=logQZToZTilde, 
                             logQZTildeToZ=logQZTildeToZ)

        
def proposeAddAndAcceptOrReject(wordType, textCorpus, hyperParameters,samplingVariables,
                                   proposalTypeProportions, numActiveTopicsForWordType):
    addedTopic = drawTopicInactiveForWordType(wordType=wordType, 
                                              activeTopics=samplingVariables.getActiveTopics(), 
                                              zMat=samplingVariables.zMat, 
                                              numActiveTopicsForWordType=numActiveTopicsForWordType)
    samplingVariables.counts.updateRevertableChangeInZ(wordType, addedTopic, 0, 1)
    
    # compute probs of moving from Z to ZTilde and vice versa
    K = len(samplingVariables.getActiveTopics())
    Ki = numActiveTopicsForWordType[wordType]
    logQZToZTilde = math.log(proposalTypeProportions[PROPOSE_ADD] / (K - Ki))
    logQZTildeToZ = math.log(proposalTypeProportions[PROPOSE_DELETE] / (Ki + 1))

    proposeAndAcceptOrReject(topic=addedTopic, 
                             isNewTopic=False, 
                             wordType=wordType,
                             textCorpus=textCorpus,
                             hyperParameters=hyperParameters, 
                             samplingVariablesWithRevertableChanges=samplingVariables,
                             proposalTypeProportions=proposalTypeProportions, 
                             numActiveTopicsForWordType=numActiveTopicsForWordType,
                             logQZToZTilde=logQZToZTilde, 
                             logQZTildeToZ=logQZTildeToZ)

def proposeDeleteAndAcceptOrReject(wordType, textCorpus, hyperParameters,samplingVariables,
                                   proposalTypeProportions, numActiveTopicsForWordType,
                                   numWordTypesActivatedInTopic):
    deletedTopic = drawTopicActiveForWordType(wordType=wordType, 
                                              activeTopics=samplingVariables.getActiveTopics(), 
                                              zMat=samplingVariables.zMat, 
                                              numActiveTopicsForWordType=numActiveTopicsForWordType)
    samplingVariables.counts.updateRevertableChangeInZ(wordType, deletedTopic, 1, 0)

    K = len(samplingVariables.getActiveTopics())
    Ki = numActiveTopicsForWordType[wordType]
    logQZToZTilde = math.log(proposalTypeProportions[PROPOSE_DELETE] / K)
    if numWordTypesActivatedInTopic[wordType] == 1:
        logQZTildeToZ = math.log(proposalTypeProportions[PROPOSE_CREATE])
    else:
        logQZTildeToZ = math.log(proposalTypeProportions[PROPOSE_ADD] / (K - Ki + 1))
    
    proposeAndAcceptOrReject(topic=deletedTopic, 
                             isNewTopic=False, 
                             wordType=wordType,
                             textCorpus=textCorpus,
                             hyperParameters=hyperParameters, 
                             samplingVariablesWithRevertableChanges=samplingVariables,
                             proposalTypeProportions=proposalTypeProportions, 
                             numActiveTopicsForWordType=numActiveTopicsForWordType,
                             logQZToZTilde=logQZToZTilde, 
                             logQZTildeToZ=logQZTildeToZ)

def drawTopicActiveForWordType(wordType, activeTopics, zMat, numActiveTopicsForWordType):
    r = random.randint(1, numActiveTopicsForWordType(wordType))
    cnt = 0
    index = -1
    while cnt < r:
        index += 1
        if zMat[wordType, activeTopics[index]]>0.9:
            cnt += 1
    return activeTopics[index]
    
def drawTopicInactiveForWordType(wordType, activeTopics, zMat, numActiveTopicsForWordType):
    r = random.randint(1, len(activeTopics) - numActiveTopicsForWordType(wordType))
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
                    excludeDocWordPositions=LQi[r+1:],
                    numWordTypesActivatedInTopics=counts.numWordTypesActivatedInTopics,
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
                    excludeDocWordPositions=LQi[r+1:],
                    numWordTypesActivatedInTopics=counts.numWordTypesActivatedInTopics,
                    numTopicOccurencesInDoc=counts.numTopicOccurencesInDoc,
                    numTopicAssignmentsToWordType=counts.numTopicAssignmentsToWordType)[drawnTopic]
        jointLogProb += logProb
    return jointLogProb


def computeLogProbZGivenWGamma(wordType, kiPlus, zMat, activeTopics, gammas, wArr, alpha,sigma,tau):
    logProbZGivenWGamma = 0.0
    for iteratingTopic in activeTopics:
        logProbZGivenWGamma += (1.0 - zMat[wordType, iteratingTopic]) \
                                * (-gammas[wordType] * wArr[iteratingTopic])
        logProbZGivenWGamma += zMat[wordType, iteratingTopic] \
                            * math.log(1.0 - math.exp(-gammas[wordType] * wArr[iteratingTopic]))
        logProbZGivenWGamma += logPoissonPsiFunc(kiPlus=kiPlus, 
                                                 wordType=wordType, 
                                                 gammas=gammas, 
                                                 alpha=alpha, 
                                                 sigma=sigma, 
                                                 tau=tau)
    
def logPoissonPsiFunc(kiPlus, wordType, gammas, alpha, sigma, tau):
        lamPoisson = expr.psiTildeFunction(t=gammas[wordType], 
                                    b=sum(gammas)-gammas[wordType],
                                    alpha=alpha,
                                    sigma=sigma,
                                    tau=tau)
        return math.log(lamPoisson**kiPlus * math.exp(-lamPoisson) / math.factorial(kiPlus))
  
    
def drawProposalTypeProportions(wordType, zMat, activeTopics):
    if getNumActiveTopicsForWordType(wordType, zMat, activeTopics) < len(activeTopics):
        return {PROPOSE_CREATE: 1.0/3, 
                PROPOSE_ADD:    1.0/3, 
                PROPOSE_DELETE: 1.0/3}
    else:
        return {PROPOSE_CREATE: 0.5, 
                PROPOSE_ADD:    0.0, 
                PROPOSE_DELETE: 0.5}
        
def drawProposalType(probs):
    vals = probs.values()
    return vals[int(np.nonzero(np.random.multinomial(1, [probs[v] for v in vals]))[0][0])]


def computeRelativeLogProbabilityForTZ(activeTopics, textCorpus, tLArr, zMat, gammas, 
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
            summand4 += gammaln(alphaF, c_f(mj) + numWordsTypeTopicCoOccurences)
        numTopicOccurences = 0
        for iteratingDocPos in range(len(textCorpus)):
            numTopicOccurences += numTopicOccurencesInDoc[iteratingDocPos]
        summand5 -= gammaln(mj / c_f(mj) + numTopicOccurences)

