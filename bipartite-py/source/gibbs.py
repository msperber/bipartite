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
import numpy as np


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

def gibbsSampler(hyperParameters, gParameters, gammas, bGraph,
                 numIterations=10000):
    #init gibbs sampler
    w = [1] * gParameters.K
    Ks=gParameters.Ks
    m=gParameters.m  
    n=gParameters.n  
    sigma=hyperParameters.sigma
    tau=hyperParameters.tau
    us = []
    uModel = []
    for i in range(gParameters.n):
        uModel.append({})
        for j in sorted(bGraph.getBooksReadByReader(i)):
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
        loglike=loglike-expr.psiFunction(sum(gammas), hyperParameters.alpha, hyperParameters.sigma, hyperParameters.tau)
        for j in range(gParameters.K):
            gammaSum=sum([gammas[i]*u[i].get(j, 0.0) for i in range(n)])
            loglike+=log(expr.kappaFunction(m[j], gammaSum, hyperParameters.alpha, hyperParameters.sigma, hyperParameters.tau))
        #contribution from w
        for j in range(gParameters.K):
            gammaSum=sum([gammas[i]*u[i].get(j, 0.0) for i in range(n)])
            log(st.gamma.pdf(w[j],m[j]-sigma,loc=0,scale=1/(tau+gammaSum)))
        print loglike 
        # TODO: likelihoods are not increasing..
    return us
    
def gibbsSamplerPGammas(hyperParameters, graphParameters, bGraph,
                 numIterations=10000):
    '''
        Gibbs sampler with parametric distribution on gammas [Caron 2012,p. 6]
    '''
    #init gibbs sampler
    assert hyperParameters.a is not None and  hyperParameters.b is not None
    
    w = [1] * graphParameters.K
    Ks=graphParameters.Ks
    m=graphParameters.m  
    n=graphParameters.n  
    #should call them w parameters?
    sigma=hyperParameters.sigma
    tau=hyperParameters.tau
    
    
    a=hyperParameters.a
    b=hyperParameters.b
    gammas= np.random.gamma(a,1.0/b,n)
    
    us = []
    uModel = []
    for i in range(n):
        uModel.append({})
        for j in sorted(bGraph.getBooksReadByReader(i)):
            uModel[i][j] = 0.5
    
    Gstar=0.0
    for _ in range(numIterations):
        u = copy.deepcopy(uModel)
        # sample u given w
        sampleUGivenGammasW(u, gammas, w, graphParameters) # sample w given u
        sampleWGivenUGammas(w, u, gammas, graphParameters, hyperParameters) # save u for prediction
        Gstar=sampleGstarGivenWGammas(hyperParameters, graphParameters,gammas)
        
        us.append(u)
        
        #calculate loglikelihood
        #contribution from U

        
        
        
        # TODO: sanity check: output likelihoods, should be increasing in most iterations 
    
    return us
    
    
def sampleGstarGivenWGammas(hyperParameters, graphParameters,gammas):
    assert hyperParameters.sigma==0
    
    return random.gammavariate(hyperParameters.alpha,1.0/(hyperParameters.tau+sum(gammas)))
        

def sampleGammasGivenGstarU(hyperParameters,gParameters, graphParameters,u,Gstar):
    a=gParmeters.a
    b=gParmeters.a

    Ks=graphParameters.Ks
    n=graphParameters.n  
    gammas = [0]*n
    for i in range(n):
        usum=sum([gammas[i]*u[i].get(j, 1.0) for j in range(K)])
        gammas[i]=random.gammavariate(a+sum(Ks),1.0/(b+usum+Gstar)) # [Caron 2012, p.6 sec 3 line 4]
        
    return gammas