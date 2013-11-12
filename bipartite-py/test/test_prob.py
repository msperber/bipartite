'''
Created on Nov 12, 2013

@author: Matthias Sperber
'''

import source.prob as prob

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
    assert False
    
def test_sampleFrom15_condition2():
    assert False
    
