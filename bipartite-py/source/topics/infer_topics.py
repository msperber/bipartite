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
from infer_topics_updates import *
from infer_topics_state import *
from infer_topics_hyperparam import *
from perplexity import *


########################
### MAIN ALGORITHM #####
########################

def conditionalPrint(string, condition):
    if condition: print string

def inferTopicsCollapsedGibbs(textCorpus, hyperParameters, numIterations, numInitialTopics=10,
                              updateHyperparameters=False, verbose=True, 
                              computeLogLikelihood = True, logLikelihoodEachIteration=True,
                              estimatePerplexityForSplitCorpus=None, pplEachIteration=False):
    
    # initialize variables
    conditionalPrint("initializing sampler..", verbose)
    samplingVariables = GibbsSamplingVariables(textCorpus=textCorpus, nTopics = numInitialTopics)
    samplingVariables.initWithFullTopicsAndGammasFromFrequencies(textCorpus, numInitialTopics)
    if estimatePerplexityForSplitCorpus is not None:
        perplexityWordAvg = {}
    if verbose:
        print "done."
    
    for iteration in range(numIterations):
        conditionalPrint("Gibbs sampling iteration: " + str(iteration), verbose)

        # actual updates:
        updateUs(textCorpus=textCorpus, samplingVariables=samplingVariables)
        updateZs(textCorpus, samplingVariables, hyperParameters)
        updateTs(textCorpus, samplingVariables, hyperParameters)
        updateWGStar(textCorpus, samplingVariables, hyperParameters)
        updateGammas(textCorpus, samplingVariables, hyperParameters)
        
        #update Hyperparameters
        if updateHyperparameters:
            sample_alpha(samplingVariables, hyperParameters)
            sample_sigma(textCorpus, samplingVariables, hyperParameters)
            sample_tau(textCorpus, samplingVariables, hyperParameters)
        
        print "num topics:", len(samplingVariables.getActiveTopics())

        # bugcheck:
        samplingVariables.counts.assertConsistency(textCorpus, samplingVariables)
        assert oneIfTopicAssignmentsSupported(textCorpus, samplingVariables.tLArr, 
                                              samplingVariables.zMat)==1
                                              
        if verbose and computeLogLikelihood and logLikelihoodEachIteration:
            print "log likelihood:", computeLogLikelihoodTWZ(
                activeTopics=samplingVariables.getActiveTopics(), 
                textCorpus=textCorpus, 
                tLArr=samplingVariables.tLArr,
                zMat=samplingVariables.zMat,
                alphaTheta=hyperParameters.alphaTheta, 
                alphaF=hyperParameters.alphaF, 
                numWordTypesActivatedInTopics=samplingVariables.counts.numWordTypesActivatedInTopic,
                numTopicOccurencesInDoc=samplingVariables.counts.numTopicOccurencesInDoc)
            
        if estimatePerplexityForSplitCorpus is not None and iteration>=0.1*numIterations:
            updatePerplexityWordAvg(perplexityWordAvg=perplexityWordAvg,
                                    iteration=iteration - 0.1*numIterations,
                                    samplingVariables=samplingVariables,
                                    hyperParameters=hyperParameters,
                                    splitCorpus=estimatePerplexityForSplitCorpus)
            if pplEachIteration and verbose:
                print "estimated perplexity:", computeTotalPerplexityFromWordAvg(perplexityWordAvg)

    
    if verbose and estimatePerplexityForSplitCorpus is not None and not pplEachIteration:
        print "estimated perplexity:", computeTotalPerplexityFromWordAvg(perplexityWordAvg)
    
    retDict = {}
    retDict['samplingVariables'] = samplingVariables
    if computeLogLikelihood:
        retDict['logLikelihood'] = computeLogLikelihoodTWZ(
                activeTopics=samplingVariables.getActiveTopics(), 
                textCorpus=textCorpus, 
                tLArr=samplingVariables.tLArr,
                zMat=samplingVariables.zMat,
                alphaTheta=hyperParameters.alphaTheta, 
                alphaF=hyperParameters.alphaF, 
                numWordTypesActivatedInTopics=samplingVariables.counts.numWordTypesActivatedInTopic,
                numTopicOccurencesInDoc=samplingVariables.counts.numTopicOccurencesInDoc)
    if estimatePerplexityForSplitCorpus is not None:
        retDict['perplexity'] = computeTotalPerplexityFromWordAvg(perplexityWordAvg)
    return retDict
        

        