'''
Created on Jan 31, 2014

@author: Matthias Sperber
'''

import numpy as np
import infer_topics_updates
import source.utility as utility
import source.prob as prob
import math
import source.expressions as expr
import random
import copy
from infer_topics_state import * # ?? not sure if good idea


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
    def __str__(self):
        return ', '.join("%s: %s" % item for item in vars(self).items())




######################## Upates for Hyperparmeters 


def sample_alpha(samplingVariables,hyperParameters, a=0.01, b=0.01): # francois chooses 0.01 and 0.01

    sum_gamma=sum(samplingVariables.gammas) + samplingVariables.gStar 
    if utility.approx_equal(hyperParameters.sigma,0.0):
        hyperParameters.alpha = np.random.gamma(a + len(samplingVariables.activeTopics), 1/(b + math.log(1 + sum_gamma/hyperParameters.tau)))
    else:
        hyperParameters.alpha = np.random.gamma(a + len(samplingVariables.activeTopics),\
                        hyperParameters.sigma /( (sum_gamma+hyperParameters.tau)**hyperParameters.sigma -hyperParameters.tau**hyperParameters.sigma))
        #alpha = np.random.gamma(a+K, sigma/( (sum_gamma + tau)^sigma - tau^sigma ) )

# according to improper prior
def sample_sigma(textCorpus, samplingVariables,hyperParameters, n_MH = 1,rw_st = .05):
    sigma=hyperParameters.sigma
    for i in range(n_MH):
        sigma_new = logistic(logit(sigma) + rw_st*np.random.normal())
        # one of these should be sigma_new?
        lograte = infer_topics_updates.computeLMarginDistribution(textCorpus=textCorpus, 
                                                  gammas=samplingVariables.gammas,
                                                  zMat=samplingVariables.zMat,
                                                  uMat=samplingVariables.uMat,
                                                  activeTopics=samplingVariables.getActiveTopics(),
                                                  alpha=hyperParameters.alpha, 
                                                  sigma=hyperParameters.sigma,
                                                  tau=hyperParameters.tau) \
            - infer_topics_updates.computeLMarginDistribution(textCorpus=textCorpus, 
                                                  gammas=samplingVariables.gammas,
                                                  zMat=samplingVariables.zMat,
                                                  uMat=samplingVariables.uMat,
                                                  activeTopics=samplingVariables.getActiveTopics(),
                                                  alpha=hyperParameters.alpha, 
                                                  sigma=hyperParameters.sigma,
                                                  tau=hyperParameters.tau)
        if np.random.rand()<math.exp(lograte):
            hyperParameters.sigma = sigma_new

def alphaThetaLogLhood(textCorpus, samplingVariables,hyperParameters):
    alphaTheta=hyperParameters.alphaTheta
    activeTopics=samplingVariables.getActiveTopics()
    
    summand1=len(textCorpus) * math.lgamma(alphaTheta)
    summand2=-len(textCorpus) *len(activeTopics)*math.lgamma(alphaTheta/len(activeTopics))
    summand3=0.0
    for iteratingDoc in range(len(textCorpus)):
        summand3 += math.lgamma(alphaTheta+len(textCorpus[iteratingDoc]))
    summand4=0.0
    for iteratingDoc in range(len(textCorpus)):
        for iteratingTopic in activeTopics:
            summand4 += math.lgamma(alphaTheta/len(activeTopics)+getNumTopicOccurencesInDoc(iteratingTopic, iteratingDoc, samplingVariables.tLArr))
    return summand1+summand2+summand3+summand4

def alphaFLogLhood(textCorpus, samplingVariables,hyperParameters):
    alphaF=hyperParameters.alphaF
    activeTopics=samplingVariables.getActiveTopics()
    
    summand1=len(textCorpus) * math.lgamma(alphaF)
    summand2=0.0
    summand3=0.0

    for iteratingTopic in activeTopics:
        mj = getNumWordTypesActivatedInTopic(iteratingTopic, samplingVariables.zMat)
        ndotdotj=getNumTopicAssignments(iteratingTopic, samplingVariables.tLArr, textCorpus) 
        summand2+=-mj*math.lgamma(alphaF/mj)-math.lgamma(alphaF+ndotdotj)
        for r in range(mj):            
            ndotijrj = 0 # ??? Fix dont know how to calculate.
            summand3=math.lgamma(alphaF/mj+ndotijrj)
    return summand1+summand2+summand3

    



def logit(p):
    return math.log(p) - math.log(1-p)


def logistic(x):
    return 1/(1+math.exp(-x))

def sample_tau(textCorpus, samplingVariables,hyperParameters, n_MH = 1,rw_st = .05):

# according to improper prior
    tau = hyperParameters.tau
    for _ in range(n_MH):
        tau_new = tau * math.exp(rw_st*np.random.normal())
        lograte = infer_topics_updates.computeLMarginDistribution(textCorpus=textCorpus, 
                                                  gammas=samplingVariables.gammas,
                                                  zMat=samplingVariables.zMat,
                                                  uMat=samplingVariables.uMat,
                                                  activeTopics=samplingVariables.getActiveTopics(),
                                                  alpha=hyperParameters.alpha, 
                                                  sigma=hyperParameters.sigma,
                                                  tau=hyperParameters.tau)+ \
            - infer_topics_updates.computeLMarginDistribution(textCorpus=textCorpus, 
                                                  gammas=samplingVariables.gammas,
                                                  zMat=samplingVariables.zMat,
                                                  uMat=samplingVariables.uMat,
                                                  activeTopics=samplingVariables.getActiveTopics(),
                                                  alpha=hyperParameters.alpha, 
                                                  sigma=hyperParameters.sigma,
                                                  tau=hyperParameters.tau)
        if random.random()<math.exp(lograte):
            hyperParameters.tau = tau_new
    