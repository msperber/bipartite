'''
Created on Dec 1, 2013

@author: Matthias Sperber
'''

from nose.tools import assert_equal, assert_raises, assert_list_equal
from source.document_data import *

def test_Document_initialized():
    doc = Document(wordCounts = {1:2, 4:9})
    assert len(doc)==11
    assert len([w for w in doc if w==1])==2
    assert len([w for w in doc if w==4])==9
    
def test_Document_empty():
    bow = Document()
    assert len(bow)==0

def test_DocumentCorpus_loadFromDatFile_docs():
    corpus = DocumentCorpus.loadFromDatFile("test_document_data_files/sample.dat")
    assert len(corpus)==3
    
    assert len(corpus[0])==3
    assert_equal(set([0,6,4]), set(corpus[0]))
    
    assert len(corpus[1])==5
    assert len([l for l in corpus[1] if l==0]) == 3
    assert len([l for l in corpus[1] if l==1]) == 1
    assert len([l for l in corpus[1] if l==2]) == 1
    
    assert len(corpus[2])==0
    assert_list_equal([], corpus[2])
    
    assert_raises(Exception, corpus.getVocabList)

def test_DocumentCorpus_loadFromDatFile_vocab():
    corpus = DocumentCorpus.loadFromDatFile("test_document_data_files/sample.dat",
                                            vocabFile="test_document_data_files/sample.vocab")
    assert_list_equal(["i", "new", "percent", "people", "year", "two", "million"],
                      corpus.getVocabList())

def test_DocumentCorpus_loadFromCorpusFile_testVocab():
    corpus = DocumentCorpus.loadFromCorpusFile("test_document_data_files/sample.corpus")
    desiredVocab = ['1618', '1929', '1988', '2', '29', '303rd', '63', 'A', "America's",
                    'Black', 'Clocks', 'Depression', 'English', 'Exchange', 'Great', 'In',
                    'London', 'New', 'Oct', 'On', 'Prices', 'Raleigh', 'Saturday', 'Sir',
                    'Stock', 'There', 'Today', "Today's", "Tuesday", 'Walter', 'York', 
                    'a.m', 'adventurer', 'amid', 'and', 'are', 'at', "back", 'began', 
                    'collapsed', 'courtier', 'date', 'day', 'daylight-saving', 'days', 
                    'descended', 'ends', 'executed', 'fall', 'highlight', 'history', 'hour',
                    'in', 'investors', 'is', 'left', 'local', 'military', 'of', 'one', 'out', 
                    'panic', 'poet', 'reminder', 'selling', 'the', 'this', 'thousands', 
                    'time', 'tomorrow', 'upon', 'was', 'were', 'wiped', 'year']
    assert_list_equal(desiredVocab,
                      corpus.getVocabList())

def test_DocumentCorpus_loadFromCorpusFile_totalNumWords():
    corpus = DocumentCorpus.loadFromCorpusFile("test_document_data_files/sample.corpus")
    assert corpus.getTotalNumWords() == 84

def test_DocumentCorpus_loadFromCorpusFile_testVocabLowercase():
    corpus = DocumentCorpus.loadFromCorpusFile("test_document_data_files/sample.corpus", lowercase=True)
    desiredVocab= ['1618', '1929', '1988', '2', '29', '303rd', '63', 'a', 'a.m', 'adventurer', 
                   "america's", 'amid', 'and', 'are', 'at', 'back', 'began', 'black', 
                   'clocks', 'collapsed', 'courtier', 'date', 'day', 'daylight-saving', 
                   'days', 'depression', 'descended', 'ends', 'english', 'exchange', 
                   'executed', 'fall', 'great', 'highlight', 'history', 'hour', 'in', 
                   'investors', 'is', 'left', 'local', 'london', 'military', 'new', 'oct', 
                   'of', 'on', 'one', 'out', 'panic', 'poet', 'prices', 'raleigh', 
                   'reminder', 'saturday', 'selling', 'sir', 'stock', 'the', 'there', 
                   'this', 'thousands', 'time', 'today', "today's", 'tomorrow', 'tuesday', 
                   'upon', 'walter', 'was', 'were', 'wiped', 'year', 'york']
    assert_list_equal(desiredVocab,
                      corpus.getVocabList())

def test_DocumentCorpus_loadFromCorpusFile_lowercase_wordFrequencies():
    corpus = DocumentCorpus.loadFromCorpusFile("test_document_data_files/sample.corpus", lowercase=True)
    desiredFrequencies = {'and': 2, 'executed': 1, 'investors': 1, 'of': 2, 'is': 1, 
                          'courtier': 1, 'history': 1, 'back': 1, 'one': 1, 'prices': 1, 
                          '63': 1, 'london': 1, 'are': 1, 'year': 1, '1929': 1, 
                          'daylight-saving': 1, 'saturday': 1, 'depression': 1, '303rd': 1, 
                          'sir': 1, "america's": 1, '1618': 1, 'wiped': 1, 'there': 1, 
                          'thousands': 1, '29': 1, '1988': 1, 'tomorrow': 1, '2': 1, 
                          'black': 1, 'time': 2, 'poet': 1, 'new': 1, 'reminder': 1, 
                          'out': 1, 'was': 1, 'tuesday': 1, 'today': 1, 'walter': 1, 
                          'ends': 1, 'selling': 1, 'exchange': 1, 'a.m': 1, 'collapsed': 1, 
                          'english': 1, 'upon': 1, 'highlight': 1, 'york': 1, 'fall': 1, 
                          'date': 1, 'panic': 1, 'day': 1, 'a': 1, 'on': 1, 'great': 1, 
                          'amid': 1, 'in': 5, 'left': 1, 'hour': 1, 'this': 1, 'local': 1, 
                          'adventurer': 1, 'days': 1, 'oct': 1, 'clocks': 1, 'descended': 1, 
                          'raleigh': 1, "today's": 1, 'were': 1, 'military': 1, 'the': 4, 
                          'stock': 1, 'began': 1, 'at': 1}
    assert len(desiredFrequencies) == len(corpus.getVocabList())
    for (i,word) in zip(range(len(corpus.getVocabList())),corpus.getVocabList()):
        assert word in desiredFrequencies
        assert corpus.getVocabFrequencies()[i] == desiredFrequencies[word]


def test_DocumentCorpus_loadFromCorpusFile_maxNumDocs():
    corpus = DocumentCorpus.loadFromCorpusFile("test_document_data_files/sample.corpus")
    assert len(corpus)==5
    corpus = DocumentCorpus.loadFromCorpusFile("test_document_data_files/sample.corpus", 
                                               maxNumDocs=1)
    assert len(corpus)==1
    corpus = DocumentCorpus.loadFromCorpusFile("test_document_data_files/sample.corpus", 
                                               maxNumDocs=10)
    assert len(corpus)==5


    
def test_DocumentCorpus_loadFromCorpusFile_testMinTokenLen():
    corpus = DocumentCorpus.loadFromCorpusFile("test_document_data_files/sample.corpus", 
                                               lowercase=True,
                                               minTokenLen = 3)
    desiredVocab= ['1618', '1929', '1988', '303rd', 'a.m', 'adventurer', 
                   "america's", 'amid', 'and', 'are', 'back', 'began', 'black', 
                   'clocks', 'collapsed', 'courtier', 'date', 'day', 'daylight-saving', 
                   'days', 'depression', 'descended', 'ends', 'english', 'exchange', 
                   'executed', 'fall', 'great', 'highlight', 'history', 'hour',  
                   'investors', 'left', 'local', 'london', 'military', 'new', 'oct', 
                   'one', 'out', 'panic', 'poet', 'prices', 'raleigh', 
                   'reminder', 'saturday', 'selling', 'sir', 'stock', 'the', 'there', 
                   'this', 'thousands', 'time', 'today', "today's", 'tomorrow', 'tuesday', 
                   'upon', 'walter', 'was', 'were', 'wiped', 'year', 'york']
    assert_list_equal(desiredVocab,
                      corpus.getVocabList())
def test_DocumentCorpus_loadFromCorpusFile_removeStopWords():
    corpus = DocumentCorpus.loadFromCorpusFile("test_document_data_files/sample.corpus", 
                                               removeStopWords = True)
    desiredVocab= ['1618', '1929', '1988', '2', '29', '303rd', '63', "America's", 'Black', 
                   'Clocks', 'Depression', 'English', 'Exchange', 'Great', 'London', 'Oct', 
                   'Prices', 'Raleigh', 'Saturday', 'Sir', 'Stock', 'Today', "Today's", 
                   'Tuesday', 'Walter', 'York', 'a.m', 'adventurer', 'amid', 'back', 'began', 
                   'collapsed', 'courtier', 'date', 'day', 'daylight-saving', 'days', 
                   'descended', 'ends', 'executed', 'fall', 'highlight', 'history', 'hour', 
                   'investors', 'left', 'local', 'military', 'panic', 'poet', 'reminder', 
                   'selling', 'thousands', 'time', 'tomorrow', 'wiped', 'year']
    assert_list_equal(desiredVocab,
                      corpus.getVocabList())

def test_DocumentCorpus_loadFromCorpusFile_minNumTokens():
    corpus = DocumentCorpus.loadFromCorpusFile("test_document_data_files/sample.corpus", 
                                               minNumTokens=11)
    assert len(corpus)==5
    corpus = DocumentCorpus.loadFromCorpusFile("test_document_data_files/sample.corpus", 
                                               minNumTokens=12)
    assert len(corpus)==4
    corpus = DocumentCorpus.loadFromCorpusFile("test_document_data_files/sample.corpus", 
                                               minNumTokens=11, removeStopWords=True)
    assert len(corpus)==4

def test_DocumentCorpus_loadFromCorpusFile_vocabSize():
    corpus = DocumentCorpus.loadFromCorpusFile("test_document_data_files/sample.corpus", 
                                               maxVocabSize=5,
                                               lowercase=True)
    desiredVocab= ['and', 'in', 'of', 'the', 'time']
    assert_list_equal(desiredVocab,
                      corpus.getVocabList())
    # documents should now read:
    #   the of in the
    #   time time
    #   in in the
    #   of and
    #   in the and in
    assert len(corpus)==5
    assert len(corpus[0])==4
    assert len(corpus[1])==2
    assert len(corpus[2])==3
    assert len(corpus[3])==2
    assert len(corpus[4])==4
    
def test_DocumentCorpus_loadFromCorpusFile_maxVocabSizeAndNumTokens():
    corpus = DocumentCorpus.loadFromCorpusFile("test_document_data_files/sample.corpus", 
                                               maxVocabSize=5,
                                               minNumTokens=3,
                                               lowercase=True)
    desiredVocab= ['and', 'in', 'of', 'the', 'time']
    assert_list_equal(desiredVocab,
                      corpus.getVocabList())
    # documents should now read:
    #   the of in the
    #   in in the
    #   in the and in
    assert len(corpus)==3
    assert len(corpus[0])==4
    assert len(corpus[1])==3
    assert len(corpus[2])==4
    
def test_DocumentCorpus_loadFromCorpusFile_getVocabSize():
    corpus = DocumentCorpus.loadFromCorpusFile("test_document_data_files/sample.corpus")
    assert corpus.getVocabSize() == 75
    corpus = DocumentCorpus.loadFromCorpusFile("test_document_data_files/sample.corpus", 
                                               lowercase=True)
    assert corpus.getVocabSize() == 74

def test_DocumentCorpus_initializeByHand_wordIndexList():
    doc1 = Document(wordIndexList = [0,1,2,4])
    doc2 = Document(wordIndexList = [0,1,2,3,5])
    doc3 = Document(wordIndexList = [0,2,3,6])
    textCorpus = DocumentCorpus(documents=[doc1,doc2,doc3])
    assert textCorpus.getVocabSize()==7
    assert len(textCorpus)==3
    assert len(textCorpus[0])==4
    assert len(textCorpus[1])==5
    assert len(textCorpus[2])==4
    
def test_DocumentCorpus_initializeByHand_wordCounts():
    doc1 = Document(wordCounts = {2:1,4:1})
    doc2 = Document(wordCounts = {2:2,3:1})
    textCorpus = DocumentCorpus(documents=[doc1,doc2])
    assert textCorpus.getVocabSize()==5
    assert len(textCorpus)==2
    assert len(textCorpus[0])==2
    assert len(textCorpus[1])==3
    
def test_DocumentCorpus_initializeByHand_wordCounts_Vocab():
    doc1 = Document(wordCounts = {2:1,4:1})
    doc2 = Document(wordCounts = {2:2,3:1})
    textCorpus = DocumentCorpus(documents=[doc1,doc2], vocab=['a','b','c','d','e'])
    assert textCorpus.getVocabSize()==5
    assert len(textCorpus)==2
    assert len(textCorpus[0])==2
    assert len(textCorpus[1])==3
    
def test_DocumentCorpus_computeSplitCorpus_lowercase():
    corpus = DocumentCorpus.loadFromCorpusFile("test_document_data_files/sample.corpus", lowercase=True)
    # words with freq >= 2: and, of, time, in, the
    corpus1, corpus2 = corpus.split(0.5)
    assert len(corpus1.getVocabList()) == 5
    assert_list_equal(corpus1.getVocabList(), corpus2.getVocabList())
    for corp in [corpus1, corpus2]:
        for wordType in range(len(corp.getVocabList())):
            wordFound = False
            for doc in corp:
                for word in doc:
                    if word==wordType:
                        wordFound = True
            assert wordFound
