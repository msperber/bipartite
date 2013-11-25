'''
Created on Nov 11, 2013

@author: Sebastian vollmer
'''

import random
import prob as prob
import expressions as expr
import copy
from math import *
import scipy.stats as st


# assumes that u is already initialised
def sampleUGivenGammasW(u,gammas,w,gParameters):
    n=gParameters.n    
    
    for i in range(n):
        for j in u[i].keys():
            u[i][j]=prob.sampleTExp1(gammas[i]*w[j])

def sampleWGivenUGammas(w,u,gammas,gParameters, simulationParams):
    sigma=simulationParams.sigma
    tau=simulationParams.tau
    K=gParameters.K
    m=gParameters.m  
    n=gParameters.n  
    for j in range(K):
        gammaSum= sum([gammas[i]*u[i].get(j, 0.0) for i in range(n)])
        w[j] = random.gammavariate(m[j]-sigma,1/(tau+gammaSum))

def gibbsSampler(hyperParameters, gParameters, gammas, sparseMatrix,
                 numIterations=10000):
    #init gibbs sampler
    w = [1] * gParameters.K
    Ks=gParmeters.Ks
    m=gParameters.m  
    n=gParameters.n  
    sigma=hyperParameters.sigma
    tau=hyperParameters.tau
    us = []
    uModel = []
    for i in range(gParameters.n):
        uModel.append({})
        for j in sparseMatrix[i]:
            uModel[i][j] = 0.5
    
    for _ in range(numIterations):
        u = copy.deepcopy(uModel)
        # sample u given w
        sampleUGivenGammasW(u, gammas, w, gParameters) # sample w given u
        sampleWGivenUGammas(w, u, gammas, gParameters, hyperParameters) # save u for prediction
        us.append(u)
        #calculate loglikelihood
        #contribution from U
        loglike=0
        for i in range(n):
            loglike=loglike+Ks[i]*gammas[i]
        loglike=loglike-psiFunction(sum(gammas),hyperParameters)
        for j in range(K):
            gammaSum=sum([gammas[i]*u[i].get(j, 0.0) for i in range(n)])
            loglike+=log(kappaFunction(m[j],gammaSum,hyperParameters))
        #contribution from w
        for j in range(K):
            gammaSum=sum([gammas[i]*u[i].get(j, 0.0) for i in range(n)])
            log(st.gamma.pdf(w[j],m[j]-sigma,loc=0,scale=1/(tau+gammaSum)))
        print loglike 
        
        
        
        # TODO: sanity check: output likelihoods, should be increasing in most iterations 
    
    return us
        