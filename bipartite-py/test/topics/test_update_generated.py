'''
Created on Mar 15, 2014

@author: Matthias Sperber
'''

from nose.tools import assert_equal, assert_raises, assert_list_equal, with_setup
import unittest

from source.topics.generate_topics import *
from source.topics.infer_topics_updates import *

class TestUpdateGenerated(unittest.TestCase):
    
    def meanNumWordTypesActivatedPerTopic(self, samplingVariables):
        return np.sum(samplingVariables.zMat / samplingVariables.zMat.shape[1])
    
    def test1(self):
        numGenerative, numSampling = 100, 100
        hyperParameters = HyperParameters(alpha=5.0, sigma=0.5, tau=1.0, alphaTheta=1.0, 
                                          alphaF=1.0, aGamma=1.0, bGamma=1.0)
        # run generative algorithm, average statistic
        baseStatistic = 0.0
        for genIteration in range(numGenerative):
            samplingVariables = BipartiteTopicGenerator().generateTopics(
                                    vocabSize=5, 
                                    numDocuments=3, 
                                    numWordsPerDocument=5, 
                                    hyperParameters=hyperParameters)
            baseStatistic += \
                    self.meanNumWordTypesActivatedPerTopic(samplingVariables) / float(numGenerative)

        # first, we need to draw w & G*, as these are not produced by the generative algorithm
        updateWGStar(samplingVariables.textCorpus, samplingVariables, hyperParameters)
        # update z many times, based on most recently generated model
        for samplingIterating in range(numSampling):
            updateZs(samplingVariables.textCorpus, samplingVariables, hyperParameters)
            
        # compute same statistic & compare
        updatedStatistic = \
                    self.meanNumWordTypesActivatedPerTopic(samplingVariables) / float(numGenerative)
        
        assert_almost_equal(baseStatistic, updatedStatistic)