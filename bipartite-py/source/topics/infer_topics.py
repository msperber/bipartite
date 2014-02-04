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


########################
### MAIN ALGORITHM #####
########################


def inferTopicsCollapsedGibbs(textCorpus, hyperParameters, numIterations, numInitialTopics=10):
    
    # initialize variables
    samplingVariables = GibbsSamplingVariables(textCorpus=textCorpus, nTopics = numInitialTopics)
    samplingVariables.initWithFullTopicsAndGammasFromFrequencies(textCorpus, numInitialTopics)
    
    for iteration in range(numIterations):
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

        # bugcheck:
        samplingVariables.counts.assertConsistency(textCorpus, samplingVariables)
        assert oneIfTopicAssignmentsSupported(textCorpus, samplingVariables.tLArr, 
                                              samplingVariables.zMat)==1
    return samplingVariables
        

        