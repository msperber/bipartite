'''
Created on Nov 11, 2013

@author: Matthias Sperber
'''

import random
import math
import copy
import numpy as np

class HyperParameters(object):
    def __init__(self, alpha, sigma, tau, gammas=None, a=None, b=None, numReaders=None):
        assert 0 <= sigma < 1.0
        self.alpha = alpha
        self.sigma = sigma
        self.tau = tau
        self.gammas = gammas
        # [Caron 2012, p.6]
        self.a = a
        self.b = b
        self.numReaders = numReaders
    def getNumReaders(self):
        if self.numReaders is not None: return self.numReaders
        elif self.gammas is not None: return len(self.gammas)
        else: return None
    @staticmethod
    def sampleGammasIfNecessary(baseHyperParams):
        """
        return a COPY of the hyper parameters object, with sampled gammas in case they are
        only specified as hyper-parameters a and b 
        """
        if baseHyperParams.gammas is not None:
            hyperParametersWithGammas = baseHyperParams
        else:
            assert baseHyperParams.a is not None and baseHyperParams.b is not None and baseHyperParams.numReaders is not None
            numReaders = baseHyperParams.numReaders
            hyperParametersWithGammas = copy.deepcopy(baseHyperParams)
            hyperParametersWithGammas.gammas = np.random.gamma(baseHyperParams.a,1.0/baseHyperParams.b,numReaders)
        return hyperParametersWithGammas
        


def flipCoin(p):
    '''
        returns True with probability p, False with probability (1-p)
    '''
    return True if random.random() < p else False

def sampleFrom15(gammas, uxjList, mj, parameters, curWordType=None):
    '''
        sample from distribution [Caron 2012, equation 15]
        
        gammas: one for each word up to and including the current one
        uxjList: u's sampled so far, for all words before the current one
        mj: num words active in current topic
        parameters: should have a 'tau' and a 'sigma' field
        curWordType: sample u belonging to this wordType; if not given, deduce from len(gammas)
    '''
    if curWordType is None:
        curWordType=len(gammas) 
        assert len(gammas) == len(uxjList)+1
    y = random.random()
    tau = parameters.tau
    sigma = parameters.sigma
    sjn = sum([gammas[i]*uxjList[i] for i in range(curWordType-1)])
    if -0.00001 < mj - sigma < 0.00001:
        return ((tau+sjn)/gammas[-1])*((1.0+(gammas[-1]/(tau+sjn)))**y - 1.0)
    else:
        return (1.0/gammas[-1])*(((tau+sjn)**(-mj+sigma)+y*((tau+sjn+gammas[-1])**(-mj+sigma)-(tau+sjn)**(-mj+sigma)))**(1.0/(-mj+sigma))-(tau+sjn))


#def sampleTExp1(lam):
#    y=random.random()
#    return -math.log(1-(1-math.exp(-lam))*y )/lam

def sampleTExp1(lam):
    return sampleRightTruncatedExponential(lam, 1.0)
    
def sampleRightTruncatedExponential(lam, a):
    """
    sample ~ rExp(lambda, a), see [Caron 2012, Section 2.5]
    use inverse CDF, see: http://www.r-bloggers.com/r-help-follow-up-truncated-exponential/
    """
    y = np.random.random()
    return -math.log(1.0-(1.0-math.exp(-a*lam))*y)/lam
