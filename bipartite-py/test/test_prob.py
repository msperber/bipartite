'''
Created on Nov 12, 2013

@author: Matthias Sperber
'''

import source.prob as prob
import source.expressions as expr
from numpy.ma.testutils import assert_almost_equal

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

def test_sampleFrom15_runsThrough():
    # for now, let's just make sure it runs without crashing..
    sigma=0.5
    alpha=0.2
    tau=0.5
    prob.sampleFrom15([0.1,0.1], [0.1], sigma, prob.HyperParameters(alpha, sigma, tau))
    
