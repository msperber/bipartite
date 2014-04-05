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
 
def proposeCreateAndAcceptOrReject(wordType, textCorpus, hyperParameters,samplingVariables,
                                   proposalTypeProportions, numActiveTopicsForWordType):
    logProbZGivenWGamma = computeLogProbZGivenWGamma(wordType=wordType, 
                                                     kiPlus=0,
                                                     zMat=samplingVariables.zMat, 
                                                     activeTopics=samplingVariables.getActiveTopics(), 
                                                     gammas=samplingVariables.gammas, 
                                                     wArr=samplingVariables.wArr, 
                                                     alpha=hyperParameters.alpha, 
                                                     sigma=hyperParameters.sigma, 
                                                     tau=hyperParameters.tau)
    logProbZGivenWGamma = computeLogProbZGivenWGamma(wordType=wordType, 
                                                     kiPlus=1,
                                                     zMat=samplingVariables.zMat, 
                                                     activeTopics=samplingVariables.getActiveTopics(), 
                                                     gammas=samplingVariables.gammas, 
                                                     wArr=samplingVariables.wArr, 
                                                     alpha=hyperParameters.alpha, 
                                                     sigma=hyperParameters.sigma, 
                                                     tau=hyperParameters.tau)
    
    # draw new topic assignment proposal: re-draw all occurrences of wordType
    LQi = samplingVariables.counts.docWordPosListForWordTypes[wordType]
    newTopic = createRevertableTopic()
#    numWordTypesActivatedInTopics = samplingVariables.counts.numWordTypesActivatedInTopic
#    numWordTypesActivatedInTopics.setRevertable(newTopic,
#                                                    numWordTypesActivatedInTopics[newTopic]+1)
#    numWordTypesActivatedInTopics.activateRevertableChanges()
    drawRevertableTopicProposalsAndUpdateCounts(LQi=LQi, 
                                                samplingVariables=samplingVariables, 
                                                hyperParameters=hyperParameters, 
                                                textCorpus=textCorpus,
                                 numWordTypesActivatedInTopics=numWordTypesActivatedInTopics)
    
    # compute logProbWoTTildeGivenZTilde and logProbWoTGivenZ
    samplingVariables.activateRevertableChanges()
    
    # compute probs of moving from Z to ZTilde and vice versa
    logQZToZTilde = math.log(proposalTypeProportions[PROPOSE_CREATE])
    logQZTildeToZ = math.log(proposalTypeProportions[PROPOSE_DELETE] \
                             * 1.0/(1.0+numActiveTopicsForWordType[wordType]))
    
    # compute probs of drawing topics t and t~

def drawRevertableTopicProposalsAndUpdateCounts(LQi, samplingVariables, hyperParameters, textCorpus,
                                 numWordTypesActivatedInTopics):
    jointLogProb = 0.0
    for r in range(len(LQi)):
        iteratingDoc, iteratingWordPos = LQi[r]
        sampledTopic, logProb = sampleTGivenZT(
                    activeTopics=samplingVariables.getActiveTopics(),
                    doc=iteratingDoc, 
                    wordPos=iteratingWordPos,
                    alphaTheta=hyperParameters.alphaTheta, 
                    alphaF=hyperParameters.alphaF,
                    textCorpus=textCorpus,
                    tLArr=samplingVariables.tLArr,
                    zMat=samplingVariables.zMat,
                    excludeDocWordPositions=LQi[r+1:],
                    numWordTypesActivatedInTopics=numWordTypesActivatedInTopics,
                    numTopicOccurencesInDoc=samplingVariables.counts.numTopicOccurencesInDoc,
                    numTopicAssignmentsToWordType=\
                                samplingVariables.counts.numTopicAssignmentsToWordType)
        jointLogProb += logProb
        samplingVariables.tLArr[iteratingDoc].setRevertable(iteratingWordPos, sampledTopic)
        samplingVariables.tLArr.activateRevertableChanges()
        # update counts
        samplingVariables.counts.numTopicAssignmentsToWordType.addRevertable(
                (textCorpus[iteratingDoc][iteratingWordPos],
                 sampledTopic),
                +1)
        samplingVariables.tLArr.activateRevertableChanges(False)
        samplingVariables.counts.numTopicAssignmentsToWordType.addRevertable(
                (textCorpus[iteratingDoc][iteratingWordPos],
                 sampledTopic),
                -1)

        samplingVariables.tLArr.activateRevertableChanges()
        samplingVariables.counts.numTopicOccurencesInDoc.addRevertable((iteratingDoc,
                                sampledTopic), + 1)
        samplingVariables.tLArr.activateRevertableChanges(False)
        samplingVariables.counts.numTopicOccurencesInDoc.addRevertable((iteratingDoc,
                                sampledTopic), - 1)
    samplingVariables.tLArr.activateRevertableChanges()
    samplingVariables.counts.numTopicOccurencesInDoc.activateRevertableChanges()
    samplingVariables.counts.numTopicAssignmentsToWordType.activateRevertableChanges()
    
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
