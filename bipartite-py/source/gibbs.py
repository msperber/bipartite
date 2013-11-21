'''
Created on Nov 11, 2013

@author: Sebastian vollmer
'''

import random
import source.prob as prob
import source.expressions as expr


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
        gammaSum= sum([u[i].get(j, 0.0) for i in range(n)])
        w[j] = random.gammavariate(m[j]-sigma,1/(tau+gammaSum))

        