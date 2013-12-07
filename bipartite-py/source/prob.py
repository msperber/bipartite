'''
Created on Nov 11, 2013

@author: Matthias Sperber
'''

import random
import math
import copy
import numpy

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
            hyperParametersWithGammas.gammas = numpy.random.gamma(baseHyperParams.a,1.0/baseHyperParams.b,numReaders)
        return hyperParametersWithGammas
        


def flipCoin(p):
    '''
        returns True with probability p, False with probability (1-p)
    '''
    return True if random.random() < p else False

def sampleFrom15(gammas, uxjList, mj, parameters):
    '''
        sample from distribution [Caron 2012, equation 15]
    '''
    assert len(gammas) == len(uxjList)+1
    y = random.random()
    tau = parameters.tau
    sigma = parameters.sigma
    sjn = sum([gammas[i]*uxjList[i] for i in range(len(gammas)-1)])
    if -0.00001 < mj - sigma < 0.00001:
        return ((tau+sjn)/gammas[-1])*((1.0+(gammas[-1]/(tau+sjn)))**y - 1.0)
    else:
        try:
            return (1.0/gammas[-1])*(((tau+sjn)**(-mj+sigma)+y*((tau+sjn+gammas[-1])**(-mj+sigma)-(tau+sjn)**(-mj+sigma)))**(1.0/(-mj+sigma))-(tau+sjn))
        except ZeroDivisionError:
            return 5 # TODO: why 5? also, this event should never occur..
            
def sampleTExp1(lam):
    y=random.random()
    return -math.log(1-(1-math.exp(-lam))*y )/lam
    
    
        