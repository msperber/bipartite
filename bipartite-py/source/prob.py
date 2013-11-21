'''
Created on Nov 11, 2013

@author: Matthias Sperber
'''

import random
import math

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
            return 5
            
def sampleTExp1(lam):
    y=random.random()
    return -math.log(1-(1-math.exp(-lam))*y )/lam
    
    
        