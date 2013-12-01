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