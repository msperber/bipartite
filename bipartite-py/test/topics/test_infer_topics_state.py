'''
Created on Dec 28, 2013

@author: Matthias Sperber
'''

from nose.tools import assert_equal, assert_raises, assert_list_equal, with_setup
import unittest
from source.topics.infer_topics import *
from source.document_data import *
from source.utility import approx_equal
from numpy.ma.testutils import assert_almost_equal
from nose.tools.nontrivial import nottest
from source.topics.infer_topics_hyperparam import HyperParameters
from matplotlib import test
from source.topics.state import RevertableSparseDict
class TopicStateTestCase (unittest.TestCase):
    
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
    
    def test_GibbsSamplingVariables_allocate_1topic(self):
        sVars = GibbsSamplingVariables(self.textCorpus1, nTopics = 1)
        assert sVars.uMat.shape == (self.textCorpus1.getVocabSize(), 1) == (7,1)
        assert sVars.zMat.shape == (self.textCorpus1.getVocabSize(), 1) == (7,1)
        assert len(sVars.tLArr) == len(self.textCorpus1) == 3
        for i in range(3):
            assert len(sVars.tLArr[i]) == len(self.textCorpus1[i]) == 4
        assert len(sVars.gammas) == self.textCorpus1.getVocabSize() == 7
        assert len(sVars.wArr) == 1
        assert sVars.gStar is None
    
    def test_GibbsSamplingVariables_allocate_4topics(self):
        sVars = GibbsSamplingVariables(self.textCorpus1, nTopics = 4)
        assert sVars.uMat.shape == (self.textCorpus1.getVocabSize(), 4) == (7,4)
        assert sVars.zMat.shape == (self.textCorpus1.getVocabSize(), 4) == (7,4)
        assert len(sVars.tLArr) == len(self.textCorpus1) == 3
        for i in range(3):
            assert len(sVars.tLArr[i]) == len(self.textCorpus1[i]) == 4
        assert len(sVars.gammas) == self.textCorpus1.getVocabSize() == 7
        assert len(sVars.wArr) == 4
        assert sVars.gStar is None
    
    def test_GibbsSamplingVariables_initialize_4topics_consistency(self):
        nTopics = 4
        vocabSize = 7
        nDocs = 3
        sVars = GibbsSamplingVariables(self.textCorpus1, nTopics = nTopics)
        sVars.initWithFullTopicsAndGammasFromFrequencies(self.textCorpus1, nTopics = nTopics)
        for i in range(vocabSize):
            for j in range(nTopics):
                assert (sVars.zMat[i,j] == 1 and 0.0 <= sVars.uMat[i,j] < 1.0) \
                    or (sVars.zMat[i,j] == 0 and approx_equal(sVars.uMat[i,j], 1.0))
        for l in range(nDocs):
            for q in range(len(self.textCorpus1[l])):
                assert 0 <= sVars.tLArr[l][q] < nTopics
                assert sVars.zMat[self.textCorpus1[l][q], sVars.tLArr[l][q]] == 1 
        for i in range(vocabSize):
            assert sVars.gammas[i] > 0.0
        for j in range(nTopics):
            assert sVars.wArr[j] > 0.0
    
    def test_GibbsSamplingVariables_initialize_4topics_gammasFromFrequencies(self):
        nTopics = 4
        sVars = GibbsSamplingVariables(self.textCorpus1, nTopics = nTopics)
        sVars.initWithFullTopicsAndGammasFromFrequencies(self.textCorpus1, nTopics = nTopics)
        assert_almost_equal(3.0/12.0, sVars.gammas[0])
        assert_almost_equal(2.0/12.0, sVars.gammas[1])
        assert_almost_equal(2.0/12.0, sVars.gammas[2])
        assert_almost_equal(2.0/12.0, sVars.gammas[3])
        assert_almost_equal(1.0/12.0, sVars.gammas[4])
        assert_almost_equal(1.0/12.0, sVars.gammas[5])
        assert_almost_equal(1.0/12.0, sVars.gammas[6])
    
    def test_GibbsSamplingVariables_initialize_4topics_fullTopics(self):
        nTopics = 4
        vocabSize = 7
        sVars = GibbsSamplingVariables(self.textCorpus1, nTopics = nTopics)
        sVars.initWithFullTopicsAndGammasFromFrequencies(self.textCorpus1, nTopics = nTopics)
        for i in range(vocabSize):
            for j in range(nTopics):
                assert (sVars.zMat[i,j] == 1 and 0.0 <= sVars.uMat[i,j] < 1.0)
    
    def test_GibbsSamplingVariables_active_Topics(self):
        nTopics = 4
        sVars = GibbsSamplingVariables(self.textCorpus1, nTopics = nTopics)
        assert len(sVars.getActiveTopics())==nTopics
        assert set(sVars.getActiveTopics()) == set(range(nTopics)) 
    
    def test_GibbsSamplingVariables_createNewTopics(self):
        sVars = GibbsSamplingVariables(self.textCorpus1, nTopics = 4)
        newTopics = sVars.createNewTopics(2)
        assert set([4,5]) == set(newTopics)
        assert len(sVars.getActiveTopics()) == 6
        assert sVars.uMat.shape == (self.textCorpus1.getVocabSize(), 6) == (7,6)
        assert sVars.zMat.shape == (self.textCorpus1.getVocabSize(), 6) == (7,6)
        assert len(sVars.wArr) == 6
    
    def test_GibbsSamplingVariables_releaseDeadTopics(self):
        sVars = GibbsSamplingVariables(self.textCorpus1, nTopics = 4)

        for l in range(len(self.textCorpus1)):
            for q in range(len(self.textCorpus1[l])):
                sVars.tLArr[l][q]=0
        
        # make topic # 2 (and only that) a dead topic:
        for i in range(sVars.zMat.shape[0]):
            for j in range(sVars.zMat.shape[1]):
                if j==2:
                    sVars.zMat[i,j] = 0
                else:
                    sVars.zMat[i,j] = 1
        
        assert len(sVars.getActiveTopics()) == sVars.zMat.shape[1]
        sVars.releaseDeadTopics()
        assert len(sVars.getActiveTopics()) == sVars.zMat.shape[1]-1
        assert_list_equal([2],sVars.createNewTopics(1)) 
        
    
    def test_getNumTopicOccurencesInDoc(self):
        sVars = GibbsSamplingVariables(self.textCorpus1, nTopics = 4)
        
        for q in range(len(self.textCorpus1[0])):
            sVars.tLArr[0][q] = 1
        assert getNumTopicOccurencesInDoc(0, 0, sVars.tLArr) == 0
        assert getNumTopicOccurencesInDoc(1, 0, sVars.tLArr) == len(self.textCorpus1[0])
        assert getNumTopicOccurencesInDoc(2, 0, sVars.tLArr) == 0
        assert getNumTopicOccurencesInDoc(3, 0, sVars.tLArr) == 0
        
        sVars.tLArr[0][1] = 0
        assert getNumTopicOccurencesInDoc(0, 0, sVars.tLArr) == 1
        assert getNumTopicOccurencesInDoc(1, 0, sVars.tLArr) == len(self.textCorpus1[0])-1
        assert getNumTopicOccurencesInDoc(2, 0, sVars.tLArr) == 0
        assert getNumTopicOccurencesInDoc(3, 0, sVars.tLArr) == 0
    
    def test_getNumTopicAssignmentsToWordType(self):
        sVars = GibbsSamplingVariables(self.textCorpus1, nTopics = 4)
        
        for l in range(len(self.textCorpus1)):
            for q in range(len(self.textCorpus1[l])):
                sVars.tLArr[l][q] = 0
        assert getNumTopicAssignmentsToWordType(0, self.textCorpus1[0][0], sVars.tLArr, self.textCorpus1) \
                == self.textCorpus1.getVocabFrequencies()[self.textCorpus1[0][0]]
        assert getNumTopicAssignmentsToWordType(1, self.textCorpus1[0][0], sVars.tLArr, self.textCorpus1) == 0
        sVars.tLArr[0][0] = 1
        assert getNumTopicAssignmentsToWordType(0, self.textCorpus1[0][0], sVars.tLArr, self.textCorpus1) \
                == self.textCorpus1.getVocabFrequencies()[self.textCorpus1[0][0]] - 1
    
    def test_getNumWordTypesActivatedInTopic(self):
        sVars = GibbsSamplingVariables(self.textCorpus1, nTopics = 4)
        sVars.initWithFullTopicsAndGammasFromFrequencies(self.textCorpus1, nTopics = 4)
        assert getNumWordTypesActivatedInTopic(0, sVars.zMat) == self.textCorpus1.getVocabSize()
    
        sVars.zMat[1,0] = 0
        assert getNumWordTypesActivatedInTopic(0, sVars.zMat) == self.textCorpus1.getVocabSize() - 1
    
    def test_getRthActiveWordTypeInTopic(self):
        sVars = GibbsSamplingVariables(self.textCorpus1, nTopics = 4)
        sVars.initWithFullTopicsAndGammasFromFrequencies(self.textCorpus1, nTopics = 4)
        assert getRthActiveWordTypeInTopic(0, 0, sVars.zMat) == 0
        assert getRthActiveWordTypeInTopic(1, 0, sVars.zMat) == 1
        assert getRthActiveWordTypeInTopic(2, 0, sVars.zMat) == 2
        assert getRthActiveWordTypeInTopic(3, 0, sVars.zMat) == 3
        assert getRthActiveWordTypeInTopic(4, 0, sVars.zMat) == 4
        assert getRthActiveWordTypeInTopic(5, 0, sVars.zMat) == 5
        assert getRthActiveWordTypeInTopic(6, 0, sVars.zMat) == 6
        assert_raises(IndexError, getRthActiveWordTypeInTopic, 7, 0, sVars.zMat)
        
        sVars.zMat[2,0] = 0
        sVars.zMat[4,0] = 0
        assert getRthActiveWordTypeInTopic(0, 0, sVars.zMat) == 0
        assert getRthActiveWordTypeInTopic(1, 0, sVars.zMat) == 1
        assert getRthActiveWordTypeInTopic(2, 0, sVars.zMat) == 3
        assert getRthActiveWordTypeInTopic(3, 0, sVars.zMat) == 5
        assert getRthActiveWordTypeInTopic(4, 0, sVars.zMat) == 6
        assert_raises(IndexError, getRthActiveWordTypeInTopic, 5, 0, sVars.zMat)
    
    def test_oneIfTopicAssignmentsSupported(self):
        sVars = GibbsSamplingVariables(self.textCorpus1, nTopics = 4) 
        sVars.initWithFullTopicsAndGammasFromFrequencies(self.textCorpus1, nTopics = 4)
        assert oneIfTopicAssignmentsSupported(self.textCorpus1, sVars.tLArr, sVars.zMat)==1
        sVars.zMat[self.textCorpus1[0][0], sVars.tLArr[0][0]] = 0
        assert oneIfTopicAssignmentsSupported(self.textCorpus1, sVars.tLArr, sVars.zMat)==0
    
class RevertableDictTestCase (unittest.TestCase):
    def test_dictFunctions(self):
        d = RevertableSparseDict()
        d[3] = 4
        d[4,5] = 2
        d["a"] = 1
        assert d[3] == 4
        assert d[4,5] == 2
        assert d["a"] == 1
    
    def test_sparseDict(self):
        d = RevertableSparseDict()
        assert d[4] == 0
        d[4,4] = 0
        d[3] = 3
        d[2] = 2
        d[2] = 0
    
    def test_revert(self):
        d = RevertableSparseDict()
        d[0] = 0
        d[1] = 1
        d.setRevertable(1, 0)
        d.setRevertable(2,1)
        assert d[0] == 0
        assert d[1] == 0
        assert d[2] == 1
        d.revert()
        assert d[0] == 0
        assert d[1] == 1
        assert d[2] == 0

    def test_revert_add(self):
        d = RevertableSparseDict()
        d[0] = 0
        d[1] = 1
        d.addRevertable(1, -1)
        d.addRevertable(2,1)
        assert d[0] == 0
        assert d[1] == 0
        assert d[2] == 1
        d.revert()
        assert d[0] == 0
        assert d[1] == 1
        assert d[2] == 0

    def test_default(self):
        d = RevertableSparseDict(defaultReturnValue=4)
        assert d[0] == 4 
    
    def test_makePermanent(self):
        d = RevertableSparseDict()
        d[0] = 0
        d[1] = 1
        d.setRevertable(1, 0)
        d.setRevertable(2,1)
        assert d[0] == 0
        assert d[1] == 0
        assert d[2] == 1
        d.makePermanent()
        d.revert()
        assert d[0] == 0
        assert d[1] == 0
        assert d[2] == 1
         
    def test_deactivate(self):
        d = RevertableSparseDict()
        d[0] = 0
        d[1] = 1
        d.setRevertable(1, 0)
        d.setRevertable(2,1)
        assert d[0] == 0
        assert d[1] == 0
        assert d[2] == 1
        d.activateRevertableChanges(False)
        assert d[0] == 0
        assert d[1] == 1
        assert d[2] == 0
        d.activateRevertableChanges(True)
        assert d[0] == 0
        assert d[1] == 0
        assert d[2] == 1

class RevertableListTestCase (unittest.TestCase):
    def test_listFunctions(self):
        l = RevertableList()
        l.append(4)
        l.append(2)
        l.append(1)
        assert l[0] == 4
        assert l[1] == 2
        assert l[2] == 1
        assert_raises(IndexError, l.__getitem__, 4)
    
    def test_revert(self):
        l = RevertableList()
        l.append(0)
        l.append(1)
        l.setRevertable(1, 0)
        l.setRevertable(0, 0)
        assert l[0] == 0
        assert l[1] == 0
        l.revert()
        assert l[0] == 0
        assert l[1] == 1

    def test_revert_add(self):
        l = RevertableList()
        l.append(0)
        l.append(1)
        l.addRevertable(1, -1)
        l.addRevertable(0, 1)
        assert l[0] == 1
        assert l[1] == 0
        l.revert()
        assert l[0] == 0
        assert l[1] == 1
    
    def test_makePermanent(self):
        l = RevertableList()
        l.append(0)
        l.append(1)
        l.setRevertable(0, 1)
        l.setRevertable(1, 0)
        l.makePermanent()
        l.revert()
        assert l[0] == 1
        assert l[1] == 0
         
    def test_deactivate(self):
        l = RevertableList()
        l.append(0)
        l.append(1)
        l.setRevertable(0, 1)
        l.setRevertable(1, 0)
        assert l[0] == 1
        assert l[1] == 0
        l.activateRevertableChanges(False)
        assert l[0] == 0
        assert l[1] == 1
        l.activateRevertableChanges(True)
        assert l[0] == 1
        assert l[1] == 0

        