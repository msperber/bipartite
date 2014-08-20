'''
Created on Mar 15, 2014

@author: Matthias Sperber
'''

from nose.tools import assert_equal, assert_raises, assert_list_equal, with_setup
import unittest

from source.topics.generate_topics import *
from source.topics.infer_topics_updates import *
from source.topics.infer_topics_updates_metropolis import *
from nose.tools.nontrivial import nottest

class TestUpdateGenerated(): # unittest.TestCase):
    
    def meanNumWordTypesActivatedPerTopic(self, samplingVariables):
        return np.sum(samplingVariables.zMat) / samplingVariables.zMat.shape[1]
    def numTopicsWithWordTypeActivated(self, samplingVariables, wordType):
        num = 0.0
        for topic in samplingVariables.activeTopics:
            num += samplingVariables.zMat[ wordType , topic ]
        return num
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
    def assert_list_almost_equal(self, l1, l2, decimal=6):
        assert len(l1)==len(l2)
        for i in range(len(l1)):
            assert_almost_equal(l1[i], l2[i], decimal)
     
    def testZUpdates(self):
        random.seed(13)
        np.random.seed(13)

        numGenerative, numSampling = 10, 100
        hyperParameters = HyperParameters(alpha=5.0, sigma=0.5, tau=1.0, alphaTheta=1.0, 
                                          alphaF=1.0, aGamma=1.0, bGamma=1.0)
        # run generative algorithm, average statistic
        # statistic will contain avg. # topics with word 0 activate, and avg total # topics
        baseStatistic, updatedStatistic = [0.0, 0.0], [0.0, 0.0]
        for genIteration in range(numGenerative):
            try:
                samplingVariables = BipartiteTopicGenerator().generateTopics(
                                        vocabSize=5, 
                                        numDocuments=3, 
                                        numWordsPerDocument=5, 
                                        hyperParameters=hyperParameters)
            except NoTopicsException:
                continue
            baseStatistic[0] += \
                    self.numTopicsWithWordTypeActivated(samplingVariables, 0) / float(numGenerative)
            baseStatistic[1] += \
                    len(samplingVariables.activeTopics) / float(numGenerative)

            # first, we need to draw w & G*, as these are not produced by the generative algorithm
            updateWGStar(samplingVariables.textCorpus, samplingVariables, hyperParameters)
            # update z many times, based on most recently generated model
            for samplingIterating in range(numSampling):
                updateZs(samplingVariables.textCorpus, samplingVariables, hyperParameters,
                         limitUpdatesToWordTypes=[0])
            
            # compute same statistic & compare
            updatedStatistic[0] += \
                        self.numTopicsWithWordTypeActivated(samplingVariables, 0) / float(numGenerative)
            updatedStatistic[1] += \
                        len(samplingVariables.activeTopics) / float(numGenerative)
        
        print "baseStatistic, updatedStatistic:", baseStatistic, updatedStatistic
        self.assert_list_almost_equal(updatedStatistic, baseStatistic, decimal=1)

    def _nottestTUpdates(self):
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
        updatedStatistics = np.zeros(numDocuments * numWordsPerDocument+1)
        for genIteration in range(numGenerative):
            samplingVariables = BipartiteTopicGenerator().generateTopics(
                                    vocabSize=vocabSize, 
                                    numDocuments=numDocuments, 
                                    numWordsPerDocument=numWordsPerDocument, 
                                    hyperParameters=hyperParameters)
            baseStatistics += \
                    self.numWordsAssignmentsToTopicHisto(samplingVariables) / float(numGenerative)

            # first, we need to draw w & G*, as these are not produced by the generative algorithm
#            updateWGStar(samplingVariables.textCorpus, samplingVariables, hyperParameters)
            # update t many times, based on most recently generated model
            for samplingIterating in range(numSampling):
                updateTs(samplingVariables.textCorpus, samplingVariables, hyperParameters)
                # compute same statistic & compare
            updatedStatistics += \
                        self.numWordsAssignmentsToTopicHisto(samplingVariables) / float(numGenerative)
        
        self.assert_list_almost_equal(baseStatistics, updatedStatistics, 0.1)

if __name__ == "__main__":
    sys.exit(TestUpdateGenerated().testZUpdates())
