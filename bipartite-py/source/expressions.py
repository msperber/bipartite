'''
Created on Nov 11, 2013

@author: Matthias Sperber
'''

import math

class HyperParameters(object):
    def __init__(self, alpha, sigma, tau):
        assert 0 <= sigma < 1.0
        self.alpha = alpha
        self.sigma = sigma
        self.tau = tau

class GraphParameters(object):
    def __init__(self,n,K,m,Ks):
        assert n >=0
        assert K >= 0
        self.n=n    # number of readers
        self.K=K    # number of books
        self.m=m    # number of times each book was read
        self.Ks=Ks # number of books each reader has read
    @staticmethod
    def deduceFromSparseGraph(sparseMatrix):
        n=len(sparseMatrix)
        # calculate K (num books) from matrix
        Ks=[]
        K=0
        for i in range(n):
            Ks.append(len(sparseMatrix[i]))
            if len(sparseMatrix[i])>0:
                K=max([K,max(sparseMatrix[i])])
        K+=1        
        # calculate m (num times each book was read)
        m=[0]*K
        for reader in sparseMatrix:
            for book in reader:
                m[book]+=1
        return GraphParameters(n,K,m)


def lambdaFunction(w, parameters):
    assert w!=0.0
    alpha = parameters.alpha
    sigma = parameters.sigma
    tau = parameters.tau
    return (alpha/math.gamma(1.0-sigma)) * (w**(-sigma-1.0)) * (math.e ** (-w*tau)) # checked sv

def psiFunction(t, parameters):
    alpha = parameters.alpha
    sigma = parameters.sigma
    tau = parameters.tau
    if -0.000001<sigma<0.000001: # careful: division by zero
        return alpha * math.log(1.0 + t / tau) # checked sv
    else:
        return (alpha/sigma) * ((t + tau)**sigma - tau**sigma) # checked sv

def psiTildeFunction(t, b, parameters):
    # sv: For GGP it follows from Equations (3) (4) and the density of the GGP that 
    return psiFunction(t, HyperParameters(parameters.alpha, parameters.sigma, parameters.tau + b))

def kappaFunction(n, z, parameters):
    alpha = parameters.alpha
    sigma = parameters.sigma
    tau = parameters.tau
    return (alpha / (z+tau)**(n-sigma)) * (math.gamma(n-sigma)/math.gamma(1.0-sigma)) # checked sv
