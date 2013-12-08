'''
Created on Dec 1, 2013

@author: matthias
'''

from nose.tools import assert_equal, assert_raises, assert_list_equal
from source.bag_of_words import *

def test_BagOfWords_initialized():
    bow = BagOfWords({1:2, 4:9})
    assert bow.numWords()==11
    assert len(bow)==2
    assert bow[1]==2
    assert bow[4]==9
    assert_equal(set([1, 4]), set(bow.getContainedWords()))
    
def test_BagOfWords_empty():
    bow = BagOfWords({})
    assert bow.numWords()==0
    assert len(bow)==0

def test_DocumentCorpus_loadFromDatFile_docs():
    corpus = DocumentCorpus.loadFromDatFile("test_bag_of_words_files/sample.dat")
    assert len(corpus)==3
    
    assert len(corpus[0])==3
    assert corpus[0].numWords()==3
    assert_equal(set([0, 6, 4]), set(corpus[0].getContainedWords()))
    assert corpus[0][0]==1
    assert corpus[0][6]==1
    assert corpus[0][4]==1
    
    assert len(corpus[1])==3
    assert corpus[1].numWords()==5
    assert corpus[1][0]==3
    assert corpus[1][1]==1
    assert corpus[1][2]==1
    
    assert len(corpus[2])==0
    assert corpus[2].numWords()==0
    
    assert_raises(Exception, corpus.getVocabList)

def test_DocumentCorpus_loadFromDatFile_vocab():
    corpus = DocumentCorpus.loadFromDatFile("test_bag_of_words_files/sample.dat",
                                            vocabFile="test_bag_of_words_files/sample.vocab")
    assert_list_equal(["i", "new", "percent", "people", "year", "two", "million"],
                      corpus.getVocabList())

def test_DocumentCorpus_loadFromCorpusFile_testVocab():
    corpus = DocumentCorpus.loadFromCorpusFile("test_bag_of_words_files/sample.corpus")
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
def test_DocumentCorpus_loadFromCorpusFile_testVocabLowercase():
    corpus = DocumentCorpus.loadFromCorpusFile("test_bag_of_words_files/sample.corpus", lowercase=True)
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

def test_DocumentCorpus_loadFromCorpusFile_maxNumDocs():
    corpus = DocumentCorpus.loadFromCorpusFile("test_bag_of_words_files/sample.corpus")
    assert len(corpus)==5
    corpus = DocumentCorpus.loadFromCorpusFile("test_bag_of_words_files/sample.corpus", 
                                               maxNumDocs=1)
    assert len(corpus)==1
    corpus = DocumentCorpus.loadFromCorpusFile("test_bag_of_words_files/sample.corpus", 
                                               maxNumDocs=10)
    assert len(corpus)==5
    
def test_DocumentCorpus_loadFromCorpusFile_testMinTokenLen():
    corpus = DocumentCorpus.loadFromCorpusFile("test_bag_of_words_files/sample.corpus", 
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
    corpus = DocumentCorpus.loadFromCorpusFile("test_bag_of_words_files/sample.corpus", 
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
    corpus = DocumentCorpus.loadFromCorpusFile("test_bag_of_words_files/sample.corpus", 
                                               minNumTokens=11)
    assert len(corpus)==5
    corpus = DocumentCorpus.loadFromCorpusFile("test_bag_of_words_files/sample.corpus", 
                                               minNumTokens=12)
    assert len(corpus)==4
    corpus = DocumentCorpus.loadFromCorpusFile("test_bag_of_words_files/sample.corpus", 
                                               minNumTokens=11, removeStopWords=True)
    assert len(corpus)==4

def test_DocumentCorpus_loadFromCorpusFile_vocabSize():
    corpus = DocumentCorpus.loadFromCorpusFile("test_bag_of_words_files/sample.corpus", 
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
    assert corpus[0].numWords()==4
    assert corpus[1].numWords()==2
    assert corpus[2].numWords()==3
    assert corpus[3].numWords()==2
    assert corpus[4].numWords()==4
    
def test_DocumentCorpus_loadFromCorpusFile_vocabSizeAndNumTokens():
    corpus = DocumentCorpus.loadFromCorpusFile("test_bag_of_words_files/sample.corpus", 
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
    assert corpus[0].numWords()==4
    assert corpus[1].numWords()==3
    assert corpus[2].numWords()==4
    
    
    
