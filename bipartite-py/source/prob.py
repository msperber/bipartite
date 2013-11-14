'''
Created on Nov 11, 2013

@author: Matthias Sperber
'''

import random

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
    tao = parameters.tao
    sigma = parameters.sigma
    sjn = sum([gammas[i]*uxjList[i] for i in range(len(gammas)-1)])
    if -0.00001 < mj - sigma < 0.00001:
        return ((tao+sjn)/gammas[-1])*((1.0+(gammas[-1]/(tao+sjn)))**y - 1.0)
    else:
        return (1.0/gammas[-1])*(((tao+sjn)**(-mj+sigma)+y*((tao+sjn+gammas[-1])**(-mj+sigma)-(tao+sjn)**(-mj+sigma)))**(1.0/mj+sigma)-(tao+sjn))
    
        