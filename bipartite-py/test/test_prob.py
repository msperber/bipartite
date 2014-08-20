'''
Created on Nov 12, 2013

@author: Matthias Sperber
'''

import source.prob as prob
import source.expressions as expr
from numpy.ma.testutils import assert_almost_equal

import sys

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
    
def test_sampleFrom15_buckets():
    sigma=0.5
    alpha=0.2
    tau=0.5
    minU = 1E10
    maxU = -1E10
    uVals = []
    mj = sigma
    gammas = [0.1,0.1]
    us = [0.1]
    numSamples = 1000000
    for i in xrange(numSamples):
        u = prob.sampleFrom15(gammas, us, mj, prob.HyperParameters(alpha, sigma, tau))
        minU = min(minU, u)
        maxU = max(maxU, u)
        uVals.append(u)
    
    numBuckets = 100
    uBuckets = [[] for i in range(numBuckets)]
    for u in uVals:
        bucketIndex = int((u - minU)/(maxU - minU)*(numBuckets-1))
        uBuckets[bucketIndex].append(u)
    uMeans = [sum(b)/len(b) for b in uBuckets]
    
    # careful: formula for conditional probability is only proportionally specified,
    # so let's estimate the proportionality factor in a first step and then compare the bins
    propFactorAvg, propFactorCnt = 0.0, 0
    for i in range(numBuckets):
        bucketProp = float(len(uBuckets[i])) / numSamples
        if us[0] > 1 or us[0] < 0:
            conditionalProbProp = 0.0
        else:
            conditionalProbProp = expr.kappaFunction(mj+1.0, uMeans[i] + gammas[-1] + gammas[0]*us[0], 
                                             alpha, sigma, tau)
            if bucketProp > 0.0001:
                propFactorAvg += conditionalProbProp / bucketProp 
                propFactorCnt += 1
    propFactorAvg /= propFactorCnt
    for i in range(numBuckets):
        bucketProp = float(len(uBuckets[i])) / numSamples
        if us[0] > 1 or us[0] < 0:
            conditionalProbProp = 0.0
        else:
            conditionalProbProp = expr.kappaFunction(mj+1.0, uMeans[i] + gammas[-1] + gammas[0]*us[0], 
                                             alpha, sigma, tau)
        conditionalProb = conditionalProbProp / propFactorAvg
        print "conditionalProb / bucketProp:", conditionalProb, bucketProp
    assert_almost_equal(conditionalProb, bucketProp, 2)
    
if __name__ == "__main__":
    sys.exit(test_sampleFrom15_buckets())    
