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


########################
### MAIN ALGORITHM #####
########################


def inferTopicsCollapsedGibbs(textCorpus, hyperParameters, numIterations, numInitialTopics=10,
                              verbose=True):
    
    # initialize variables
    if verbose:
        print "initializing sampler.."
    samplingVariables = GibbsSamplingVariables(textCorpus=textCorpus, nTopics = numInitialTopics)
    samplingVariables.initWithFullTopicsAndGammasFromFrequencies(textCorpus, numInitialTopics)
    if verbose:
        print "done."
    
    for iteration in range(numIterations):
        if verbose:
            print "Gibbs sampling iteration:", iteration
#        print "gammas:", samplingVariables.gammas
#        print "w's:", samplingVariables.wArr
#        print "u's:", samplingVariables.uMat

        # actual updates:
        updateUs(textCorpus=textCorpus, samplingVariables=samplingVariables)
        updateZs(textCorpus, samplingVariables, hyperParameters)
        updateTs(textCorpus, samplingVariables, hyperParameters)
        updateWGStar(textCorpus, samplingVariables, hyperParameters)
        updateGammas(textCorpus, samplingVariables, hyperParameters)
        
        #update Hyperparameters
#        sample_alpha(samplingVariables, hyperParameters)
#        sample_sigma(samplingVariables, hyperParameters)
#        sample_tau(samplingVariables, hyperParameters)

        # bugcheck:
        samplingVariables.counts.assertConsistency(textCorpus, samplingVariables)
        assert oneIfTopicAssignmentsSupported(textCorpus, samplingVariables.tLArr, 
                                              samplingVariables.zMat)==1
                                              
        if verbose:
            print "log likelihood:", computeLogLikelihoodTWZ(
                activeTopics=samplingVariables.getActiveTopics(), 
                textCorpus=textCorpus, 
                tLArr=samplingVariables.tLArr,
                zMat=samplingVariables.zMat,
                alphaTheta=hyperParameters.alphaTheta, 
                alphaF=hyperParameters.alphaF, 
                numWordTypesActivatedInTopics=samplingVariables.counts.numWordTypesActivatedInTopic,
                numTopicOccurencesInDoc=samplingVariables.counts.numTopicOccurencesInDoc)
    return samplingVariables
        

        