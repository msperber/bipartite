'''
Created on Nov 12, 2013

@author: Matthias Sperber
'''

import source.prob as prob
import source.expressions as expr
from numpy.ma.testutils import assert_almost_equal
import math

def test_flipCoin_boundaries():
    heads = 0
    for _ in range(100):
        if prob.flipCoin(1.0):
            heads += 1
    assert heads == 100
    heads = 0
    for _ in range(100):
        if prob.flipCoin(0.0):
            heads += 1
    assert heads == 0

def test_flipCoin_unbiased():
    heads = 0
    for _ in range(10000):
        if prob.flipCoin(0.5):
            heads += 1
    assert 4000 < heads < 6000

def test_sampleFrom15_condition1():
#    sigma=0.5
#    alpha=0.2
#    tau=0.5
#    sum = 0.0
#    nSamples = 10000
#    for _ in range(nSamples):
#        sum += prob.sampleFrom15([0.1,0.1], [0.1], sigma, expr.Parameters(alpha, sigma, tau))
#    assert_almost_equal(math.log(), 
#                        sum/nSamples)
    assert False
    
def test_sampleFrom15_condition2():
    assert False
    
