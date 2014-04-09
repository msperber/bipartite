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

def updateZs(textCorpus, samplingVariables, hyperParameters, limitUpdatesToWordTypes=None):
    """
    a Metropolis algorithm to update zMat's and tLArr's simultaneously 
    """
    for iteratingWordType in range(textCorpus.getVocabSize()):
        if limitUpdatesToWordTypes is not None and iteratingWordType not in limitUpdatesToWordTypes:
            continue
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
        logprob1 = computeRelativeLogProbabilityForTZ(
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
                        numTopicAssignmentsToWordType=samplingVariables.counts.numTopicAssignmentsToWordType,
                        topicsMightDie=True)
    else:
        logprob1 = computeRelativeLogProbabilityForTZ(
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
                        numTopicAssignmentsToWordType=samplingVariables.counts.numTopicAssignmentsToWordType,
                        topicsMightDie=False)

    # change back to the original state
    samplingVariables.tLArr.activateRevertableChanges(False)
    numWordTypesActivatedInTopics.activateRevertableChanges(False)
    samplingVariables.counts.numTopicOccurencesInDoc.activateRevertableChanges(False)
    samplingVariables.counts.numTopicAssignmentsToWordType.activateRevertableChanges(False)
    samplingVariables.zMat[iteratingWordType, iteratingTopic] = 1-zTilde_ij
    if topicCouldBeDying:
        logprob2 = computeRelativeLogProbabilityForTZ(
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
                    numTopicAssignmentsToWordType=samplingVariables.counts.numTopicAssignmentsToWordType,
                    topicsMightDie=True)
    else:
        logprob2 = computeRelativeLogProbabilityForTZ(
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
                    numTopicAssignmentsToWordType=samplingVariables.counts.numTopicAssignmentsToWordType,
                    topicsMightDie=False)
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
def sampleTruncatedNumNewTopicsLog(activeTopics, textCorpus, tLArr, alphaTheta, wordType,
                                gammas, alpha, sigma, tau, numTopicOccurencesInDoc,
                                c_theta=expr.c_theta_K, cutoff=20):
    
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
    
    print "numNewTopics:", numNewTopics
    newTopics = samplingVariables.createNewTopics(numNewTopics)
    samplingVariables.counts.numActiveTopicsForWordType[iteratingWordType] += numNewTopics
#    print "nr new topics", numNewTopics
    wordFreqs = textCorpus.getVocabFrequencies()
    totalNumWords = textCorpus.getTotalNumWords()

    for newTopic in newTopics:
        samplingVariables.counts.numWordTypesActivatedInTopic[newTopic] = \
                samplingVariables.counts.numWordTypesActivatedInTopic.get(newTopic,0) + 1

        samplingVariables.wArr[newTopic] = 1.0
        # this seems like a bug:
#        for iteratingWordType2 in range(samplingVariables.uMat.shape[0]):
#            samplingVariables.gammas[iteratingWordType2] = \
#                    float(wordFreqs[iteratingWordType2]) / float(totalNumWords) 
        
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
