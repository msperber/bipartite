'''
Created on Dec 1, 2013

@author: Matthias Sperber
'''

class BagOfWords(dict):
    def __init__(self, wordCounts):
        self.update(wordCounts)
    def numWords(self):
        return sum([self[i] for i in self.keys()])
    def getContainedWords(self):
        return self.keys()
        
class DocumentCorpus(list):
    """
    a corpus is a list of bags of words (access documents like with a normal python list)
    """
    def __init__(self, documents=[], vocab=None):
        self.extend(documents)
        self.vocab = vocab
    def getVocabList(self):
        if self.vocab is None:
            raise Exception("no vocab was specified")
        else:
            return self.vocab
    
    def computeSplitCorpus(self, ratio):
        """
        splits the corpus vertically into two: each new corpus will have the same
        list of documents as before, but words in each document will be split randomly
        between both topics, according to the given ratio   
        """
        # TODO: implement
        raise NotImplementedError()
    @staticmethod
    def loadFromDatFile(datFile, vocabFile=None):
        """
        assumes this format:
        3 1:2 5:1
        1 4:1
        
        optionally, a corresponding vocab file can be specified
        """
        vocab = None
        if vocabFile is not None:
            vocab = [line.strip() for line in open(vocabFile).readlines()]
        
        documents=[]
        for line in open(datFile):
            spl = line.split()
            assert len(spl)>=1
            numWords = int(spl[0])
            wordCounts = {}
            for wordCountStr in spl[1:]:
                wordIndex, count = tuple([int(c) for c in wordCountStr.split(":")])
                assert count>0
                wordCounts[wordIndex] = count
            bagOfWords = BagOfWords(wordCounts)
            assert numWords==bagOfWords.numWords()
            documents.append(bagOfWords)
        return DocumentCorpus(documents=documents, vocab=vocab)
    
    @staticmethod
    def loadFromTextFile():
        # TODO: implement
        raise NotImplementedError()
