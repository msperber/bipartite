'''
Created on Mar 15, 2014

@author: Matthias Sperber
'''

from nose.tools import assert_equal, assert_raises, assert_list_equal, with_setup
import unittest

from source.topics.generate_topics import *
from source.topics.infer_topics_updates import *
from nose.tools.nontrivial import nottest

class TestUpdateGenerated(unittest.TestCase):
    
    def meanNumWordTypesActivatedPerTopic(self, samplingVariables):
        return np.sum(samplingVariables.zMat / samplingVariables.zMat.shape[1])
    def numWordsAssignmentsToTopicHisto(self, samplingVariables):
        histo = np.zeros(samplingVariables.textCorpus.getTotalNumWords()+1)
        wordsAssignedToTopicSets = [list() for _ in range(len(samplingVariables.getActiveTopics()))]
        for docPos in range(len(samplingVariables.textCorpus)):
            for wordPos in range(len(samplingVariables.textCorpus[docPos])):
                wordsAssignedToTopicSets[samplingVariables.tLArr[docPos][wordPos]]\
                        .append(samplingVariables.textCorpus[docPos][wordPos])
        for i in range(len(wordsAssignedToTopicSets)):
            histo[len(wordsAssignedToTopicSets[i])] += 1
        assert_almost_equal(samplingVariables.textCorpus.getTotalNumWords(),
                            sum([histo[i]*i for i in range(len(histo))]))
        return histo
    def assert_list_almost_equal(self, l1, l2, precision=1e-6):
        assert len(l1)==len(l2)
        for i in range(len(l1)):
            assert_almost_equal(l1[i], l2[i], precision)
     
    @nottest
    def ljhjtestZUpdates(self):
        random.seed(13)
        np.random.seed(13)

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

    def testTUpdates(self):
        random.seed(13)
        np.random.seed(13)

        numGenerative, numSampling = 100, 1000
        hyperParameters = HyperParameters(alpha=5.0, sigma=0.5, tau=1.0, alphaTheta=1.0, 
                                          alphaF=1.0, aGamma=1.0, bGamma=1.0)
        vocabSize=5 
        numDocuments=3
        numWordsPerDocument=5
        # run generative algorithm, average statistic
        baseStatistics = np.zeros(numDocuments * numWordsPerDocument+1)
        for genIteration in range(numGenerative):
            samplingVariables = BipartiteTopicGenerator().generateTopics(
                                    vocabSize=vocabSize, 
                                    numDocuments=numDocuments, 
                                    numWordsPerDocument=numWordsPerDocument, 
                                    hyperParameters=hyperParameters)
            baseStatistics += \
                    self.numWordsAssignmentsToTopicHisto(samplingVariables) / float(numGenerative)

        # first, we need to draw w & G*, as these are not produced by the generative algorithm
        updateWGStar(samplingVariables.textCorpus, samplingVariables, hyperParameters)
        # update t many times, based on most recently generated model
        updatedStatistics = np.zeros(numDocuments * numWordsPerDocument+1)
        for samplingIterating in range(numSampling):
            updateTs(samplingVariables.textCorpus, samplingVariables, hyperParameters)
            # compute same statistic & compare
            updatedStatistics += \
                        self.numWordsAssignmentsToTopicHisto(samplingVariables) / float(numSampling)
        updatedStatisticsFinalState = \
                        self.numWordsAssignmentsToTopicHisto(samplingVariables)   
        
        self.assert_list_almost_equal(baseStatistics, updatedStatistics, 0.1)