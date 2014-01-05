'''
Created on Dec 28, 2013

@author: Matthias Sperber
'''

from nose.tools import assert_equal, assert_raises, assert_list_equal, with_setup
from source.topics.infer_topics import *
from source.document_data import *
from source.utility import approx_equal
from numpy.ma.testutils import assert_almost_equal
from nltk.sem.logic import IllegalTypeException
from nose.tools.nontrivial import nottest

doc1 = Document(wordIndexList = [0,1,2,4])
doc2 = Document(wordIndexList = [0,1,3,5])
doc3 = Document(wordIndexList = [0,2,3,6])
textCorpus1 = DocumentCorpus(documents=[doc1,doc2,doc3])

def setup_func():
    random.seed(13)
    np.random.seed(13)


def test_GibbsSamplingVariables_allocate_1topic():
    sVars = GibbsSamplingVariables(textCorpus1, nTopics = 1)
    assert sVars.uMat.shape == (textCorpus1.getVocabSize(), 1) == (7,1)
    assert sVars.zMat.shape == (textCorpus1.getVocabSize(), 1) == (7,1)
    assert len(sVars.tLArr) == len(textCorpus1) == 3
    for i in range(3):
        assert len(sVars.tLArr[i]) == len(textCorpus1[i]) == 4
    assert len(sVars.gammas) == textCorpus1.getVocabSize() == 7
    assert len(sVars.wArr) == 1
    assert sVars.gStar is None

def test_GibbsSamplingVariables_allocate_4topics():
    sVars = GibbsSamplingVariables(textCorpus1, nTopics = 4)
    assert sVars.uMat.shape == (textCorpus1.getVocabSize(), 4) == (7,4)
    assert sVars.zMat.shape == (textCorpus1.getVocabSize(), 4) == (7,4)
    assert len(sVars.tLArr) == len(textCorpus1) == 3
    for i in range(3):
        assert len(sVars.tLArr[i]) == len(textCorpus1[i]) == 4
    assert len(sVars.gammas) == textCorpus1.getVocabSize() == 7
    assert len(sVars.wArr) == 4
    assert sVars.gStar is None

def test_GibbsSamplingVariables_initialize_4topics_consistency():
    nTopics = 4
    vocabSize = 7
    nDocs = 3
    sVars = GibbsSamplingVariables(textCorpus1, nTopics = nTopics)
    for i in range(vocabSize):
        for j in range(nTopics):
            assert (sVars.zMat[i,j] == 1 and 0.0 <= sVars.uMat[i,j] < 1.0) \
                or (sVars.zMat[i,j] == 0 and approx_equal(sVars.uMat[i,j], 1.0))
    for l in range(nDocs):
        for q in range(len(textCorpus1[l])):
            assert 0 <= sVars.tLArr[l][q] < nTopics
            assert sVars.zMat[textCorpus1[l][q], sVars.tLArr[l][q]] == 1 
    for i in range(vocabSize):
        assert sVars.gammas[i] > 0.0
    for j in range(nTopics):
        assert sVars.wArr[j] > 0.0

def test_GibbsSamplingVariables_initialize_4topics_gammasFromFrequencies():
    nTopics = 4
    sVars = GibbsSamplingVariables(textCorpus1, nTopics = nTopics)
    assert_almost_equal(3.0/12.0, sVars.gammas[0])
    assert_almost_equal(2.0/12.0, sVars.gammas[1])
    assert_almost_equal(2.0/12.0, sVars.gammas[2])
    assert_almost_equal(2.0/12.0, sVars.gammas[3])
    assert_almost_equal(1.0/12.0, sVars.gammas[4])
    assert_almost_equal(1.0/12.0, sVars.gammas[5])
    assert_almost_equal(1.0/12.0, sVars.gammas[6])

def test_GibbsSamplingVariables_initialize_4topics_fullTopics():
    nTopics = 4
    vocabSize = 7
    sVars = GibbsSamplingVariables(textCorpus1, nTopics = nTopics)
    for i in range(vocabSize):
        for j in range(nTopics):
            assert (sVars.zMat[i,j] == 1 and 0.0 <= sVars.uMat[i,j] < 1.0)

def test_GibbsSamplingVariables_active_Topics():
    nTopics = 4
    sVars = GibbsSamplingVariables(textCorpus1, nTopics = nTopics)
    assert len(sVars.getActiveTopics())==nTopics
    assert set(sVars.getActiveTopics()) == set(range(nTopics)) 

def test_GibbsSamplingVariables_createNewTopics():
    sVars = GibbsSamplingVariables(textCorpus1, nTopics = 4)
    newTopics = sVars.createNewTopics(2)
    assert set([4,5]) == set(newTopics)
    assert len(sVars.getActiveTopics()) == 6
    assert sVars.uMat.shape == (textCorpus1.getVocabSize(), 6) == (7,6)
    assert sVars.zMat.shape == (textCorpus1.getVocabSize(), 6) == (7,6)
    assert len(sVars.wArr) == 6

def test_GibbsSamplingVariables_releaseDeadTopics():
    sVars = GibbsSamplingVariables(textCorpus1, nTopics = 4)
    
    # make topic # 2 (and only that) a dead topic:
    prevTopic = 0
    for l in range(len(textCorpus1)):
        for q in range(len(textCorpus1[l])):
            if prevTopic==2:
                sVars.tLArr[l][q] = 3
            else:
                sVars.tLArr[l][q] = prevTopic
            prevTopic = (prevTopic+1) % 3
    
    assert len(sVars.getActiveTopics()) == 4
    sVars.releaseDeadTopics()
    assert len(sVars.getActiveTopics()) == 3
    assert_list_equal([2],sVars.createNewTopics(1)) 
    

def test_getNumTopicOccurencesInDoc():
    sVars = GibbsSamplingVariables(textCorpus1, nTopics = 4)
    
    for q in range(len(textCorpus1[0])):
        sVars.tLArr[0][q] = 1
    assert getNumTopicOccurencesInDoc(0, 0, sVars.tLArr) == 0
    assert getNumTopicOccurencesInDoc(1, 0, sVars.tLArr) == len(textCorpus1[0])
    assert getNumTopicOccurencesInDoc(2, 0, sVars.tLArr) == 0
    assert getNumTopicOccurencesInDoc(3, 0, sVars.tLArr) == 0
    
    sVars.tLArr[0][1] = 0
    assert getNumTopicOccurencesInDoc(0, 0, sVars.tLArr) == 1
    assert getNumTopicOccurencesInDoc(1, 0, sVars.tLArr) == len(textCorpus1[0])-1
    assert getNumTopicOccurencesInDoc(2, 0, sVars.tLArr) == 0
    assert getNumTopicOccurencesInDoc(3, 0, sVars.tLArr) == 0

def test_getNumTopicAssignmentsToWordType():
    sVars = GibbsSamplingVariables(textCorpus1, nTopics = 4)
    
    for l in range(len(textCorpus1)):
        for q in range(len(textCorpus1[l])):
            sVars.tLArr[l][q] = 0
    assert getNumTopicAssignmentsToWordType(0, textCorpus1[0][0], sVars.tLArr, textCorpus1) \
            == textCorpus1.getVocabFrequencies()[textCorpus1[0][0]]
    assert getNumTopicAssignmentsToWordType(1, textCorpus1[0][0], sVars.tLArr, textCorpus1) == 0
    sVars.tLArr[0][0] = 1
    assert getNumTopicAssignmentsToWordType(0, textCorpus1[0][0], sVars.tLArr, textCorpus1) \
            == textCorpus1.getVocabFrequencies()[textCorpus1[0][0]] - 1

def test_getNumWordTypesActivatedInTopic():
    sVars = GibbsSamplingVariables(textCorpus1, nTopics = 4)
    
    assert getNumWordTypesActivatedInTopic(0, sVars.zMat) == textCorpus1.getVocabSize()

    sVars.zMat[1,0] = 0
    assert getNumWordTypesActivatedInTopic(0, sVars.zMat) == textCorpus1.getVocabSize() - 1

def test_getRthActiveWordTypeInTopic():
    sVars = GibbsSamplingVariables(textCorpus1, nTopics = 4)
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

def test_oneIfTopicAssignmentsSupported():
    sVars = GibbsSamplingVariables(textCorpus1, nTopics = 4)
    assert oneIfTopicAssignmentsSupported(textCorpus1, sVars.tLArr, sVars.zMat)==1
    sVars.zMat[textCorpus1[0][0], sVars.tLArr[0][0]] = 0
    assert oneIfTopicAssignmentsSupported(textCorpus1, sVars.tLArr, sVars.zMat)==0

@with_setup(setup_func)
def test_inferTopicsCollapsedGibbs_runsWithoutException():
    hyperParameters = HyperParameters(alpha=5.0, sigma=0.5, tau=1.0, alphaTheta=1.0, alphaF=1.0)
    inferTopicsCollapsedGibbs(textCorpus1, hyperParameters, 10)