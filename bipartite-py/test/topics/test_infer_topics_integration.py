'''
Created on Dec 28, 2013

@author: Matthias Sperber
'''

from nose.tools import assert_equal, assert_raises, assert_list_equal, with_setup
import unittest
from source.topics.infer_topics import *
from source.document_data import *
from nose.tools.nontrivial import nottest
from source.topics.infer_topics_hyperparam import HyperParameters

class TopicIntegrationTestCase (unittest.TestCase):
    
    doc1a = Document(wordIndexList = [0,1,2,4])
    doc2a = Document(wordIndexList = [0,1,3,5])
    doc3a = Document(wordIndexList = [0,2,3,6])
    textCorpus1=DocumentCorpus(documents=[doc1a,doc2a,doc3a])

    doc1b = Document(wordIndexList = [1,2,4])
    doc2b = Document(wordIndexList = [1,3,5])
    doc3b = Document(wordIndexList = [0,3,6])
    testCorpus=DocumentCorpus(documents=[doc1b,doc2b,doc3b])
   
    def seedRandomGeneratorsDeterministically(self):
        random.seed(13)
        np.random.seed(13)
 
    @nottest
    def test_inferTopicsCollapsedGibbs_runsWithoutException(self):
        self.seedRandomGeneratorsDeterministically()
        hyperParameters = HyperParameters(alpha=5.0, sigma=0.5, tau=1.0, alphaTheta=1.0, 
                                          alphaF=1.0, aGamma=1.0, bGamma=1.0)
        sv = inferTopicsCollapsedGibbs(self.textCorpus1, hyperParameters, numIterations=200, 
                                       numInitialTopics=1, verbose=True,
                                       estimatePerplexityForSplitCorpus=self.testCorpus)
        print "final t matrix: ", sv.tLArr
        assert False
