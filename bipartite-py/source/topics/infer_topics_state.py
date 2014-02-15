'''
Created on Jan 24, 2014

@author: Matthias Sperber
'''

import numpy as np
import source.utility as utility
import source.prob as prob
import math
import source.expressions as expr
import random
import copy

class GibbsSamplingVariables(object):
    def __init__(self, textCorpus, nTopics = 1):
        self.deadTopics, self.activeTopics = [], []
        self.textCorpus = textCorpus
        self.allocateVars(textCorpus, nTopics)
        self.initWithFullTopicsAndGammasFromFrequencies(textCorpus, nTopics)
        self.counts = None
        
    def allocateVars(self, textCorpus, nTopics):
        vocabSize = textCorpus.getVocabSize()
        
        # scores for word-types in topics:
        self.uMat = np.empty((vocabSize, nTopics)) 
        
        # which word-types belong to which topics:
        self.zMat = np.empty((vocabSize, nTopics), dtype=np.int8)

        # topic assignments
        self.tLArr = []
        
        for doc in textCorpus:
            self.tLArr.append(np.empty(len(doc)))
            
        # reading interest ("word popularity")
        self.gammas = np.empty((vocabSize,))
        
        # topic popularity
        self.wArr = np.empty((nTopics,))
        
        self.gStar = None
        
        self.activeTopics = range(nTopics)
        
    def initWithFullTopicsAndGammasFromFrequencies(self, textCorpus, nTopics):
        # initialize variables to a consistent state
        # ensure cosistency by making all words belong to all topics initially
        self.uMat.fill(0.5)
        self.zMat.fill(1)
        for iteratingDoc in range(len(textCorpus)):
            self.tLArr[iteratingDoc] = np.random.randint(0, 
                                                         nTopics, 
                                                         len(self.tLArr[iteratingDoc]))
        wordFreqs = textCorpus.getVocabFrequencies()
        totalNumWords = textCorpus.getTotalNumWords()
        for iteratingWordType in range(textCorpus.getVocabSize()):
            self.gammas[iteratingWordType] = \
                        float(wordFreqs[iteratingWordType]) / float(totalNumWords)
        self.wArr.fill(1.0)
        self.counts = GibbsCounts(textCorpus, self)
    
    # approach to managing active & dead topics: both are stored in (complementary) lists,
    # which are only changed upon a call of releaseDeadTopics() or createNewTopics()
    # thus, topics with no associated words remain "active" until releaseDeadTopics() gets called
    
    def getActiveTopics(self):
        return self.activeTopics
    
    def releaseDeadTopics(self):
        removingTopics = []
        for topic in self.activeTopics:
            if getNumWordTypesActivatedInTopic(topic, self.zMat)==0:
                removingTopics.append(topic)
        for topic in removingTopics:
            self.activeTopics.remove(topic)
        self.deadTopics.extend(removingTopics)
        return removingTopics
    
    def createNewTopics(self, numNewTopics):
        # to be on the safe side: init new zMat's with 1, new uMat's with 0.5, new wArr with 1
        newTopics = []
        for _ in range(numNewTopics):
            if len(self.deadTopics)==0:
                newTopic = len(self.activeTopics)
                # expand zMat
                newZ = np.ones((self.zMat.shape[0], self.zMat.shape[1]+1))
                newZ[:,:-1] = self.zMat
                self.zMat = newZ
                # expand uMat
                newU = np.empty((self.uMat.shape[0], self.uMat.shape[1]+1))
                for i in range(self.uMat.shape[0]):
                    newU[i,-1] = 0.5
                newU[:,:-1] = self.uMat
                self.uMat = newU
                # expand wArr
                newW = np.ones((self.wArr.shape[0]+1))
                newW[:-1] = self.wArr
                self.wArr = newW
            else:
                newTopic = self.deadTopics.pop()
                for i in range(self.zMat.shape[0]):
                    self.zMat[i,newTopic] = 1
                    self.uMat[i,newTopic] = 0.5
                self.wArr[newTopic] = 1.0
            newTopics.append(newTopic)
            self.activeTopics.append(newTopic)
            
        return newTopics
    
    def removeTopic(self, topic):
        self.deadTopics.append(topic)
        self.activeTopics.remove(topic)
        
class RevertableSparseDict(dict):
    """
    * revertable: can make temporary changes and revert these changes later
    * sparse: when key is unknown, value 0 will be returned 
    ** note: len() and keys() only reflect the non-revertable part, while getitem reflects the revertable part
    """ 
    def __init__(self, defaultReturnValue=0, *args, **kw):
        super(RevertableSparseDict,self).__init__(*args, **kw)
        self.tmpDict = {}
        self.activateRevertable=True
        self.defaultReturnValue=defaultReturnValue
    def setRevertable(self, key, value):
        self.tmpDict[key] = value
    def addRevertable(self, key, value):
        if key in self.tmpDict:
            self.tmpDict[key] += value
        else:
            self.tmpDict[key] = self.__getitem__(key) + value
    def revert(self):
        self.tmpDict.clear()
    def makePermanent(self):
        for k in self.tmpDict:
            self.__setitem__(k, self.tmpDict[k])
        self.tmpDict.clear()
    def __getitem__(self, key):
        if self.activateRevertable and key in self.tmpDict:
            return self.tmpDict[key]
        elif key in super(RevertableSparseDict, self).keys():
            return super(RevertableSparseDict, self).__getitem__(key)
        return self.defaultReturnValue
    def activateRevertableChanges(self, value=True):
        self.activateRevertable=value
    
class GibbsCounts(object):
    """
    Counts are implemented as pair-indexed dictionaries (for now).
    If the dictionary has no entry for some key, the corresponding value is meant to be 0 by 
    convention.
    All updates are made from the outside, the class itself gives no guarantees as for consistency.
    """
    def __init__(self, textCorpus, samplingVariables):
        self.numTopicOccurencesInDoc = RevertableSparseDict()
        self.numTopicAssignmentsToWordType = RevertableSparseDict()
        self.numWordTypesActivatedInTopic = RevertableSparseDict()
        self.numActiveTopicsForWordType = RevertableSparseDict()
        self.docWordPosListForTopicAssignments = RevertableSparseDict(defaultReturnValue=[])
        for topic in samplingVariables.getActiveTopics():
            for docId in range(len(textCorpus)):
                self.numTopicOccurencesInDoc[docId, topic] = \
                        getNumTopicOccurencesInDoc(topic=topic, doc=docId, 
                                                   tLArr=samplingVariables.tLArr)
            for wordType in range(textCorpus.getVocabSize()):
                self.numTopicAssignmentsToWordType[wordType, topic] = \
                        getNumTopicAssignmentsToWordType(topic=topic, wordType=wordType,
                                                          tLArr=samplingVariables.tLArr,
                                                          textCorpus=textCorpus)
            self.numWordTypesActivatedInTopic[topic] = \
                    getNumWordTypesActivatedInTopic(topic=topic, zMat=samplingVariables.zMat)
        for wordType in range(textCorpus.getVocabSize()):
            self.numActiveTopicsForWordType[wordType] = \
                    getNumActiveTopicsForWordType(wordType=wordType, zMat=samplingVariables.zMat, 
                                          activeTopics=samplingVariables.getActiveTopics())
        for docId in range(len(textCorpus)):
            for wordPos in range(len(textCorpus[docId])):
                wordType = textCorpus[docId][wordPos]
                assignedTopic = samplingVariables.tLArr[docId][wordPos]
                if (wordType, assignedTopic) not in self.docWordPosListForTopicAssignments:
                    self.docWordPosListForTopicAssignments[wordType, assignedTopic] = []
                self.docWordPosListForTopicAssignments[wordType, assignedTopic].append((docId,wordPos))
    def assertConsistency(self, textCorpus, samplingVariables):
        for topic in samplingVariables.getActiveTopics():
            for docId in range(len(textCorpus)):
                assert self.numTopicOccurencesInDoc.get((docId, topic),0) == \
                        getNumTopicOccurencesInDoc(topic=topic, doc=docId, 
                                                   tLArr=samplingVariables.tLArr)
            for wordType in range(textCorpus.getVocabSize()):
                assert self.numTopicAssignmentsToWordType.get((wordType, topic),0) == \
                        getNumTopicAssignmentsToWordType(topic=topic, wordType=wordType,
                                                          tLArr=samplingVariables.tLArr,
                                                          textCorpus=textCorpus)
            assert self.numWordTypesActivatedInTopic[topic] == \
                    getNumWordTypesActivatedInTopic(topic=topic, zMat=samplingVariables.zMat)
        for wordType in range(textCorpus.getVocabSize()):
            assert self.numActiveTopicsForWordType[wordType] == \
                    getNumActiveTopicsForWordType(wordType=wordType, zMat=samplingVariables.zMat, 
                                          activeTopics=samplingVariables.getActiveTopics())
    @staticmethod
    def getNumTopicAssignmentsToWordTypeExcl(wordType, topic, tLArr, textCorpus,
                                     numTopicAssignmentsToWordTypeDict, excludeDocWordPositions):
        exclSum = 0
        # make sure we don't exclude a word twice for some reason:
        excludeDocWordPositions = list(set(excludeDocWordPositions))
        for (doc, iteratingWordPos) in excludeDocWordPositions:
            if tLArr[doc][iteratingWordPos]==topic and textCorpus[doc][iteratingWordPos]==wordType:
                exclSum += 1
        return numTopicAssignmentsToWordTypeDict.get((wordType,topic),0) - exclSum

def getNumTopicOccurencesInDoc(topic, doc, tLArr,
                                    excludeDocWordPositions=[]):
    num = 0
    for iteratingWordPos in range(len(tLArr[doc])):
        if tLArr[doc][iteratingWordPos]==topic:
            if (doc, iteratingWordPos) not in excludeDocWordPositions:
                num += 1
    return num

def getNumTopicAssignmentsToWordType(topic, wordType, tLArr, textCorpus, 
                                    excludeDocWordPositions=[]):
    num = 0
    for iteratingDoc in range(len(tLArr)):
        for iteratingWordPos in range(len(tLArr[iteratingDoc])):
            if topic==tLArr[iteratingDoc][iteratingWordPos] \
                    and textCorpus[iteratingDoc][iteratingWordPos]==wordType:
                if (iteratingDoc, iteratingWordPos) not in excludeDocWordPositions:
                    num += 1
    return num

def getNumTopicAssignments(topic, tLArr, textCorpus, # sjv
                                    excludeDocWordPositions=[]):
    num = 0
    for iteratingDoc in range(len(tLArr)):
        for iteratingWordPos in range(len(tLArr[iteratingDoc])):
            if topic==tLArr[iteratingDoc][iteratingWordPos]:
                if (iteratingDoc, iteratingWordPos) not in excludeDocWordPositions:
                    num += 1
    return num

def getNumWordTypesActivatedInTopic(topic, zMat):
    return int(np.asscalar(zMat[:,topic].sum()))

def getRthActiveWordTypeInTopic(r, topic, zMat):
    return np.nonzero(zMat[:,topic])[0][r]

def getNumActiveTopicsForWordType(wordType, zMat, activeTopics):
    n = 0
    for j in activeTopics:
        n += zMat[wordType, j]
    return int(n)

def getDocWordPosListOfWordTypeAssignedToTopic(wordType, topic, tLArr, textCorpus):
    LQij = []
    for iteratingDoc in range(len(tLArr)):
        for iteratingWordPos in range(len(tLArr[iteratingDoc])):
            if textCorpus[iteratingDoc][iteratingWordPos]==wordType \
                    and tLArr[iteratingDoc][iteratingWordPos]==topic:
                LQij.append((iteratingDoc, iteratingWordPos))
    return LQij