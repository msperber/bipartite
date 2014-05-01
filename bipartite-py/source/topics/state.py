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


class RevertableParent(object):
    def getRevertableObjects(self):
        pass
    def activateRevertableChanges(self, value=True):
        for obj in self.getRevertableObjects():
            obj.activateRevertableChanges(value)
    def revert(self):
        for obj in self.getRevertableObjects():
            obj.revert()
    def makePermanent(self):
        for obj in self.getRevertableObjects():
            obj.makePermanent()

class GibbsSamplingVariables(RevertableParent):
    
    def __init__(self, textCorpus, nTopics = 1, vocabSize=None):
        self.deadTopics, self.activeTopics = [], []
        self.textCorpus = textCorpus
        self.allocateVars(textCorpus, nTopics, vocabSize=vocabSize)
        self.counts = None
        self.revertableZChange = None
        self.revertableUChange = None
        self.revertableWChange = None
        
    def allocateVars(self, textCorpus, nTopics, vocabSize=None):
        if vocabSize is None:
            vocabSize = textCorpus.getVocabSize()
        
        # scores for word-types in topics:
        self.uMat = np.empty((vocabSize, nTopics)) 
        
        # which word-types belong to which topics:
        self.zMat = np.empty((vocabSize, nTopics), dtype=np.int8)

        # topic assignments
        self.tLArr = RevertableListList()
        
        for doc in textCorpus:
            self.tLArr.append(RevertableList(np.empty(len(doc))))
            
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
            self.tLArr[iteratingDoc] = RevertableList(np.random.randint(0, 
                                                         nTopics, 
                                                         len(self.tLArr[iteratingDoc])))
        wordFreqs = textCorpus.getVocabFrequencies()
        totalNumWords = textCorpus.getTotalNumWords()
        for iteratingWordType in range(textCorpus.getVocabSize()):
            self.gammas[iteratingWordType] = \
                        float(wordFreqs[iteratingWordType]) / float(totalNumWords)
        self.wArr.fill(1.0)
        self.initCounts(textCorpus)
    
    def initCounts(self, textCorpus):
        self.counts = GibbsCounts(textCorpus, self)
    
    def getRevertableObjects(self):
        return [self.counts, self.tLArr]
    def activateRevertableChanges(self, value=True):
        if self.revertableZChange is not None:
            if value:
                self.zMat[self.revertableZChange[0], self.revertableZChange[1]] = self.revertableZChange[3]
            else:
                self.zMat[self.revertableZChange[0], self.revertableZChange[1]] = self.revertableZChange[2]
        if self.revertableUChange is not None:
            if value:
                self.uMat[self.revertableUChange[0], self.revertableUChange[1]] = self.revertableUChange[3]
            else:
                self.uMat[self.revertableUChange[0], self.revertableUChange[1]] = self.revertableUChange[2]
        if self.revertableWChange is not None:
            if value:
                self.wArr[self.revertableWChange[0]] = self.revertableUChange[2]
            else:
                self.wArr[self.revertableUChange[0]] = self.revertableUChange[1]
        for obj in self.getRevertableObjects():
            obj.activateRevertableChanges(value)
    def revert(self):
        if self.revertableZChange is not None:
            self.zMat[self.revertableZChange[0], self.revertableZChange[1]] = self.revertableZChange[2]
            self.revertableZChange = None
        if self.revertableUChange is not None:
            self.uMat[self.revertableUChange[0], self.revertableUChange[1]] = self.revertableUChange[2]
            self.revertableUChange = None
        if self.revertableWChange is not None:
            self.wArr[self.revertableWChange[0]] = self.revertableWChange[1]
            self.revertableWChange = None
        for obj in self.getRevertableObjects():
            obj.revert()
    def makePermanent(self):
        if self.revertableZChange is not None:
            self.zMat[self.revertableZChange[0], self.revertableZChange[1]] = self.revertableZChange[3]
            self.revertableZChange = None
        if self.revertableUChange is not None:
            self.uMat[self.revertableUChange[0], self.revertableUChange[1]] = self.revertableUChange[3]
            self.revertableUChange = None
        if self.revertableWChange is not None:
            self.wArr[self.revertableWChange[0]] = self.revertableWChange[2]
            self.revertableWChange = None
        for obj in self.getRevertableObjects():
            obj.makePermanent()
    
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
        if len(removingTopics)>0: print "removingTopics:", removingTopics
        for topic in removingTopics:
            self.activeTopics.remove(topic)
        self.deadTopics.extend(removingTopics)
        return removingTopics
    def revertableChangeInZ(self, wordType, topic, oldZ, newZ):
        self.counts.updateRevertableChangeInZ(wordType, topic, oldZ, newZ)
        self.revertableZChange = (wordType, topic, oldZ, newZ)
        self.zMat[wordType,topic] = newZ
    def revertableChangeInU(self, wordType, topic, oldU, newU):
        self.revertableUChange = (wordType, topic, oldU, newU)
        self.uMat[wordType,topic] = newU
    def revertableChangeInW(self, topic, oldW, newW):
        self.revertableWChange = (topic, oldW, newW)
        self.wArr[topic] = newW
    def preparePotentialNewTopic(self, activeForWord):
        newTopic=None
        if len(self.deadTopics)==0:
            newTopic = len(self.activeTopics)
            self.deadTopics.append(newTopic)
            # expand zMat
            newZ = np.zeros((self.zMat.shape[0], self.zMat.shape[1]+1))
            newZ[0, activeForWord] = 1
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
            newTopic = self.deadTopics[0]
            for i in range(self.zMat.shape[0]):
                self.zMat[i,newTopic] = 0
                self.uMat[i,newTopic] = 0.5
            self.zMat[activeForWord,newTopic] = 1
            self.wArr[newTopic] = 1.0
        return newTopic
    
    def createNewTopic(self, preparedPotentialTopicIndex=None):
        # to be on the safe side: init new zMat's with 1, new uMat's with 0.5, new wArr with 1
        if len(self.deadTopics)==0 and preparedPotentialTopicIndex is None:
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
            if preparedPotentialTopicIndex is None:
                newTopic = self.deadTopics.pop()
                for i in range(self.zMat.shape[0]):
                    self.zMat[i,newTopic] = 1
                    self.uMat[i,newTopic] = 0.5
                self.wArr[newTopic] = 1.0
            else:
                newTopic = preparedPotentialTopicIndex
                self.deadTopics.remove(newTopic)
        self.activeTopics.append(newTopic)
        return newTopic
        
    
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
        if self.activateRevertable and len(self.tmpDict)>0 and key in self.tmpDict:
            return self.tmpDict[key]
        elif key in super(RevertableSparseDict, self).keys():
            return super(RevertableSparseDict, self).__getitem__(key)
        return self.defaultReturnValue
    def activateRevertableChanges(self, value=True):
        self.activateRevertable=value
        
class RevertableListList(list):
    def revert(self):
        for i in self:
            i.revert()
    def makePermanent(self):
        for i in self:
            i.makePermanent()
    def activateRevertableChanges(self, value=True):
        for i in self:
            i.activateRevertableChanges(value)

class RevertableList(list):
    """
    * revertable: can make temporary changes and revert these changes later
    ** note: revertable actions only change behavior of __getitem__(), not of iterating etc.
    """ 
    def __init__(self, *args, **kw):
        super(RevertableList,self).__init__(*args, **kw)
        self.tmpDict = {}
        self.activateRevertable=True
    def setRevertable(self, index, value):
        self.tmpDict[index] = value
    def addRevertable(self, index, value):
        if index in self.tmpDict:
            self.tmpDict[index] += value
        else:
            self.tmpDict[index] = self.__getitem__(index) + value
    def revert(self):
        self.tmpDict.clear()
    def makePermanent(self):
        for k in self.tmpDict:
            self.__setitem__(k, self.tmpDict[k])
        self.tmpDict.clear()
    def __getitem__(self, index):
        if self.activateRevertable and len(self.tmpDict)>0 and index in self.tmpDict:
            return self.tmpDict[index]
        else:
            return super(RevertableList, self).__getitem__(index)
    def activateRevertableChanges(self, value=True):
        self.activateRevertable=value
    
class GibbsCounts(RevertableParent):
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
        self.docWordPosListForWordTypes = RevertableSparseDict(defaultReturnValue=[])
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
                if wordType not in self.docWordPosListForWordTypes:
                    self.docWordPosListForWordTypes[wordType] = []
                self.docWordPosListForWordTypes[wordType].append((docId,wordPos))
    def updateChangeInT(self, docPos, wordPos, wordType, oldTopic, newTopic):
        self.numTopicAssignmentsToWordType[ \
                                wordType,
                                oldTopic] \
                        = self.numTopicAssignmentsToWordType.get(( \
                                wordType,
                                oldTopic),0) - 1
        self.numTopicAssignmentsToWordType[ \
                            wordType,
                            newTopic] \
                    = self.numTopicAssignmentsToWordType.get(( \
                            wordType,
                            newTopic), 0) + 1
        self.numTopicOccurencesInDoc[docPos,oldTopic] = \
                self.numTopicOccurencesInDoc.get((docPos,oldTopic),0) \
                -1
        self.numTopicOccurencesInDoc[docPos,newTopic] = \
                self.numTopicOccurencesInDoc.get((docPos,newTopic),0) \
                +1
    def updateRevertableChangeInT(self, docPos, wordPos, wordType, oldTopic, newTopic):
        self.numTopicAssignmentsToWordType.addRevertable(
                (wordType,
                 newTopic),
                +1)
        self.numTopicAssignmentsToWordType.addRevertable(
                (wordType,
                 oldTopic),
                -1)

        self.numTopicOccurencesInDoc.addRevertable((docPos, newTopic), + 1)
        self.numTopicOccurencesInDoc.addRevertable((docPos, oldTopic), - 1)
        
    def updateChangeInZ(self, wordType, topic, oldVal, newVal):
        if newVal==1:
            self.numWordTypesActivatedInTopic[topic]=\
                                                self.numWordTypesActivatedInTopic[topic]+1
            self.numActiveTopicsForWordType[wordType]=\
                                                self.numActiveTopicsForWordType[wordType]+1
        else: 
            self.numWordTypesActivatedInTopic[topic]=\
                                                    self.numWordTypesActivatedInTopic[topic]-1
            self.numActiveTopicsForWordType[wordType]=\
                                                self.numActiveTopicsForWordType[wordType]-1
    def updateRevertableChangeInZ(self, wordType, topic, oldVal, newVal):
        if newVal==1:
            self.numWordTypesActivatedInTopic.setRevertable(topic,
                                                self.numWordTypesActivatedInTopic[topic]+1)
            self.numActiveTopicsForWordType.setRevertable(wordType,
                                                self.numActiveTopicsForWordType[wordType]+1)
        else: 
            self.numWordTypesActivatedInTopic.setRevertable(topic,
                                                    self.numWordTypesActivatedInTopic[topic]-1)
            self.numActiveTopicsForWordType.setRevertable(wordType,
                                                self.numActiveTopicsForWordType[wordType]-1)
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
    def getRevertableObjects(self):
        return [self.numTopicOccurencesInDoc, self.numTopicAssignmentsToWordType,
                self.numWordTypesActivatedInTopic, self.numActiveTopicsForWordType,
                self.docWordPosListForWordTypes]

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

class RevertableParent(object):
    def getRevertableObjects(self):
        pass
    def activateRevertableChanges(self, value=True):
        for obj in self.getRevertableObjects():
            obj.activateRevertableChanges(value)
    def revert(self):
        for obj in self.getRevertableObjects():
            obj.revert()
    def makePermanent(self):
        for obj in self.getRevertableObjects():
            obj.makePermanent()

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