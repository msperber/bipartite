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

def sampleFrom15():
    '''
        samples from distribution [Caron 2012, equation 15]
    '''
    # TODO: implement