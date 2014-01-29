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

class HyperParameters(object):
    def __init__(self, alpha, sigma, tau, alphaTheta, alphaF, aGamma, bGamma):
        self.alphaTheta = alphaTheta
        self.alphaF = alphaF
        assert 0 <= sigma < 1.0
        self.alpha = alpha
        self.sigma = sigma
        self.tau = tau
        self.aGamma = aGamma
        self.bGamma = bGamma




######################## Upates for Hyperparmeters 


def sample_alpha(samplingVariables,hyperParameters, a=0.01, b=0.01): # francois chooses 0.01 and 0.01

    sum_gamma=sum(samplingVariables.gammas) + samplingVariables.gStar 
    if utility.approx_equal(hyperParameters.sigma,0.0):
        hyperParameters.alpha = np.random.gamma(a + len(samplingVariables.activeTopics), 1/(b + log(1 + sum_gamma/hyperParameters.tau)))
    else:
        hyperParameters.alpha = np.random.gamma(a + len(samplingVariables.activeTopics),\
                        hyperParameters.sigma /( (sum_gamma+hyperParameters.tau)**hyperParameters.sigma -hyperParameters.tau**hyperParameters.sigma))
        #alpha = np.random.gamma(a+K, sigma/( (sum_gamma + tau)^sigma - tau^sigma ) )

# according to improper prior
def sample_sigma(samplingVariables,hyperParameters, n_MH = 1,rw_st = .05):
    sigma=hyperParameters.sigma
    for i in range(n_MH):
        sigma_new = logistic(logit(sigma) + rw_st*np.random.normal())
        lograte = computeLMarginDistribution(samplingVariables,hyperParameters)+ \
        - computeLMarginDistribution(samplingVariables,hyperParameters)
        if np.random.rand()<exp(lograte):
            hyperParameters.sigma = sigma_new




def logit(p):
    return log(p) - log(1-p)


def logistic(x):
    return 1/(1+exp(-x))

def sample_tau(samplingVariables,hyperParameters, n_MH = 1,rw_st = .05):

# according to improper prior
    tau = hyperParameters.tau
    for i in range(n_MH):
        tau_new = tau * exp(rw_st*np.random.normal())
        lograte =computeLMarginDistribution(samplingVariables,hyperParameters)+ \
            - computeLMarginDistribution(samplingVariables,hyperParameters)
        if rand<exp(lograte):
            hyperParameters.tau = tau_new
    
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
        updateWGStar(textCorpus, samplingVariables, hyperParameters)
        updateGammas(textCorpus, samplingVariables, hyperParameters)
        
        #update Hyperparameters

        # bugcheck:
        samplingVariables.counts.assertConsistency(textCorpus, samplingVariables)
        assert oneIfTopicAssignmentsSupported(textCorpus, samplingVariables.tLArr, 
                                              samplingVariables.zMat)==1
    return samplingVariables
        

        