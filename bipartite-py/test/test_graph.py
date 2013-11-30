'''
Created on Nov 30, 2013

@author: Matthias Sperber
'''

from source.graph import *
from nose.tools import assert_raises, assert_list_equal, assert_equal, assert_almost_equal

def test_SparseBinaryBipartiteGraph_getActiveReaders_finite():
    g = SparseBinaryBipartiteGraph(numReaders=3, numBooks=5)
    assert_equal(set(), g.getActiveReaders())
    g.readBook(0, 3)
    assert_equal(set([0]), g.getActiveReaders())
    g.readBook(2, 2)
    assert_equal(set([0,2]), g.getActiveReaders())
    g.readBook(2, 4)
    assert_equal(set([0,2]), g.getActiveReaders())
def test_SparseBinaryBipartiteGraph_getActiveReaders_infinite():
    g = SparseBinaryBipartiteGraph()
    assert_equal(set(), g.getActiveReaders())
    g.readBook(0, 3)
    assert_equal(set([0]), g.getActiveReaders())
    g.readBook(2, 2)
    assert_equal(set([0,2]), g.getActiveReaders())
    g.readBook(2, 4)
    assert_equal(set([0,2]), g.getActiveReaders())
    
def test_SparseBinaryBipartiteGraph_getNumActiveReaders_finite():
    g = SparseBinaryBipartiteGraph(numReaders=5)
    assert g.getNumActiveReaders()==0
    g.readBook(3, 4)
    assert g.getNumActiveReaders()==1
def test_SparseBinaryBipartiteGraph_getNumActiveReaders_infinite():
    g = SparseBinaryBipartiteGraph()
    assert g.getNumActiveReaders()==0
    g.readBook(3, 4)
    assert g.getNumActiveReaders()==1

def test_SparseBinaryBipartiteGraph_getActiveBooks_finite():
    g = SparseBinaryBipartiteGraph(numReaders=3, numBooks=5)
    assert_equal(set(), g.getActiveBooks())
    g.readBook(0, 3)
    assert_equal(set([3]), g.getActiveBooks())
    g.readBook(2, 2)
    assert_equal(set([2,3]), g.getActiveBooks())
    g.readBook(0, 2)
    assert_equal(set([2,3]), g.getActiveBooks())
def test_SparseBinaryBipartiteGraph_getActiveBooks_infinite():
    g = SparseBinaryBipartiteGraph()
    assert_equal(set(), g.getActiveBooks())
    g.readBook(0, 3)
    assert_equal(set([3]), g.getActiveBooks())
    g.readBook(2, 2)
    assert_equal(set([2,3]), g.getActiveBooks())
    g.readBook(0, 2)
    assert_equal(set([2,3]), g.getActiveBooks())
    
def test_SparseBinaryBipartiteGraph_getNumActiveBooks_finite():
    g = SparseBinaryBipartiteGraph(numBooks=5)
    assert g.getNumActiveBooks()==0
    g.readBook(3, 3)
    assert g.getNumActiveBooks()==1
def test_SparseBinaryBipartiteGraph_getNumActiveBooks_infinite():
    g = SparseBinaryBipartiteGraph()
    assert g.getNumActiveBooks()==0
    g.readBook(3, 4)
    assert g.getNumActiveBooks()==1
    
def test_SparseBinaryBipartiteGraph_getMaxNumReaders_finite():
    g = SparseBinaryBipartiteGraph(numReaders=5)
    assert_equal(g.getMaxNumReaders(), 5)
def test_SparseBinaryBipartiteGraph_getMaxNumReaders_infinite():    
    g = SparseBinaryBipartiteGraph()
    assert_equal(g.getMaxNumReaders(), float("inf"))

def test_SparseBinaryBipartiteGraph_getMaxNumBooks_finite():
    g = SparseBinaryBipartiteGraph(numBooks=10)
    assert_equal(g.getMaxNumBooks(), 10)
def test_SparseBinaryBipartiteGraph_getMaxNumBooks_infinite():    
    g = SparseBinaryBipartiteGraph()
    assert_equal(g.getMaxNumBooks(), float("inf"))

def test_SparseBinaryBipartiteGraph_getBooksReadByReader_infinite_empty():
    g = SparseBinaryBipartiteGraph()
    g.readBook(3, 3)
    assert_equal(set(), g.getBooksReadByReader(5))
def test_SparseBinaryBipartiteGraph_getBooksReadByReader_infinite_nonempty():
    g = SparseBinaryBipartiteGraph()
    g.readBook(3, 3)
    g.readBook(3, 1)
    assert_equal(set([1,3]), g.getBooksReadByReader(3))
def test_SparseBinaryBipartiteGraph_getBooksReadByReader_finite_invalid():
    g = SparseBinaryBipartiteGraph(numReaders=5)
    g.readBook(3, 3)
    assert_raises(IndexError, g.getBooksReadByReader, 5)
def test_SparseBinaryBipartiteGraph_getBooksReadByReader_finite_empty():
    g = SparseBinaryBipartiteGraph(numReaders=6)
    g.readBook(3, 3)
    assert_equal(set(), g.getBooksReadByReader(5))
def test_SparseBinaryBipartiteGraph_getBooksReadByReader_finite_nonempty():
    g = SparseBinaryBipartiteGraph(numReaders=6, numBooks=10)
    g.readBook(3, 3)
    g.readBook(3, 4)
    assert_equal(set([3,4]), g.getBooksReadByReader(3))

def test_SparseBinaryBipartiteGraph_getReadersOfBook_infinite_empty():
    g = SparseBinaryBipartiteGraph()
    g.readBook(3, 3)
    assert_equal(set(), g.getReadersOfBook(5))
def test_SparseBinaryBipartiteGraph_getReadersOfBook_infinite_nonempty():
    g = SparseBinaryBipartiteGraph()
    g.readBook(3, 1)
    g.readBook(1, 1)
    g.readBook(1, 2)
    assert_equal(set([1,3]), g.getReadersOfBook(1))
def test_SparseBinaryBipartiteGraph_getReadersOfBook_finite_invalid():
    g = SparseBinaryBipartiteGraph(numBooks=5)
    assert_raises(IndexError, g.getReadersOfBook, 5)
def test_SparseBinaryBipartiteGraph_getReadersOfBook_finite_empty():
    g = SparseBinaryBipartiteGraph(numBooks=6)
    g.readBook(3, 3)
    assert_equal(set(), g.getReadersOfBook(5))
def test_SparseBinaryBipartiteGraph_getReadersOfBook_finite_nonempty():
    g = SparseBinaryBipartiteGraph(numReaders=6, numBooks=10)
    g.readBook(3, 3)
    g.readBook(4, 3)
    g.readBook(4, 2)
    assert_equal(set([3,4]), g.getReadersOfBook(3))

def test_SparseBinaryBipartiteGraph_readBook_finite_valid():
    g = SparseBinaryBipartiteGraph(numReaders=6, numBooks=10)
    assert not g.isReaderOfBook(3,5)
    g.readBook(3, 5)
    assert g.isReaderOfBook(3,5)
    assert not g.isReaderOfBook(3,0)

def test_SparseBinaryBipartiteGraph_readBook_finite_invalid():
    g = SparseBinaryBipartiteGraph(numReaders=6, numBooks=10)
    assert_raises(IndexError, g.readBook, 6, 5)
    assert_raises(IndexError, g.readBook, 3, 10)

def test_SparseBinaryBipartiteGraph_readBook_infinite():
    g = SparseBinaryBipartiteGraph()
    assert not g.isReaderOfBook(1003,1005)
    g.readBook(1003, 1005)
    assert g.isReaderOfBook(1003,1005)
    assert not g.isReaderOfBook(1003,1000)

def test_SparseBinaryBipartiteGraph_isReaderOfBook_finite_invalid():
    g = SparseBinaryBipartiteGraph(numReaders=6, numBooks=10)
    assert_raises(IndexError, g.isReaderOfBook, 6, 5)

def test_SparseBinaryBipartiteGraph_isReaderOfBook_infinite_valid():
    g = SparseBinaryBipartiteGraph()
    assert not g.isReaderOfBook(6, 5)
    
######################################

def test_SparseScoredBipartiteGraph_getActiveReaders_finite():
    g = SparseScoredBipartiteGraph(numReaders=3, numBooks=5)
    assert_equal(set(), g.getActiveReaders())
    g.readBook(0, 3, 1.5)
    assert_equal(set([0]), g.getActiveReaders())
    g.readBook(2, 2, 1.5)
    assert_equal(set([0,2]), g.getActiveReaders())
    g.readBook(2, 4, 1.5)
    assert_equal(set([0,2]), g.getActiveReaders())
def test_SparseScoredBipartiteGraph_getActiveReaders_infinite():
    g = SparseScoredBipartiteGraph()
    assert_equal(set(), g.getActiveReaders())
    g.readBook(0, 3, 1.5)
    assert_equal(set([0]), g.getActiveReaders())
    g.readBook(2, 2, 1.5)
    assert_equal(set([0,2]), g.getActiveReaders())
    g.readBook(2, 4, 1.5)
    assert_equal(set([0,2]), g.getActiveReaders())
    
def test_SparseScoredBipartiteGraph_getNumActiveReaders_finite():
    g = SparseScoredBipartiteGraph(numReaders=5)
    assert g.getNumActiveReaders()==0
    g.readBook(3, 4, 1.5)
    assert g.getNumActiveReaders()==1
def test_SparseScoredBipartiteGraph_getNumActiveReaders_infinite():
    g = SparseScoredBipartiteGraph()
    assert g.getNumActiveReaders()==0
    g.readBook(3, 4, 1.5)
    assert g.getNumActiveReaders()==1

def test_SparseScoredBipartiteGraph_getActiveBooks_finite():
    g = SparseScoredBipartiteGraph(numReaders=3, numBooks=5)
    assert_equal(set(), g.getActiveBooks())
    g.readBook(0, 3, 1.5)
    assert_equal(set([3]), g.getActiveBooks())
    g.readBook(2, 2, 1.5)
    assert_equal(set([2,3]), g.getActiveBooks())
    g.readBook(0, 2, 1.5)
    assert_equal(set([2,3]), g.getActiveBooks())
def test_SparseScoredBipartiteGraph_getActiveBooks_infinite():
    g = SparseScoredBipartiteGraph()
    assert_equal(set(), g.getActiveBooks())
    g.readBook(0, 3, 1.5)
    assert_equal(set([3]), g.getActiveBooks())
    g.readBook(2, 2, 1.5)
    assert_equal(set([2,3]), g.getActiveBooks())
    g.readBook(0, 2, 1.5)
    assert_equal(set([2,3]), g.getActiveBooks())
    
def test_SparseScoredBipartiteGraph_getNumActiveBooks_finite():
    g = SparseScoredBipartiteGraph(numBooks=5)
    assert g.getNumActiveBooks()==0
    g.readBook(3, 3, 1.5)
    assert g.getNumActiveBooks()==1
def test_SparseScoredBipartiteGraph_getNumActiveBooks_infinite():
    g = SparseScoredBipartiteGraph()
    assert g.getNumActiveBooks()==0
    g.readBook(3, 4, 1.5)
    assert g.getNumActiveBooks()==1
    
def test_SparseScoredBipartiteGraph_getMaxNumReaders_finite():
    g = SparseScoredBipartiteGraph(numReaders=5)
    assert_equal(g.getMaxNumReaders(), 5)
def test_SparseScoredBipartiteGraph_getMaxNumReaders_infinite():    
    g = SparseScoredBipartiteGraph()
    assert_equal(g.getMaxNumReaders(), float("inf"))

def test_SparseScoredBipartiteGraph_getMaxNumBooks_finite():
    g = SparseScoredBipartiteGraph(numBooks=10)
    assert_equal(g.getMaxNumBooks(), 10)
def test_SparseScoredBipartiteGraph_getMaxNumBooks_infinite():    
    g = SparseScoredBipartiteGraph()
    assert_equal(g.getMaxNumBooks(), float("inf"))

def test_SparseScoredBipartiteGraph_getBooksReadByReader_infinite_empty():
    g = SparseScoredBipartiteGraph()
    g.readBook(3, 3, 1.5)
    assert_equal(set(), g.getBooksReadByReader(5))
def test_SparseScoredBipartiteGraph_getBooksReadByReader_infinite_nonempty():
    g = SparseScoredBipartiteGraph()
    g.readBook(3, 3, 1.5)
    g.readBook(3, 1, 1.5)
    assert_equal(set([1,3]), g.getBooksReadByReader(3))
def test_SparseScoredBipartiteGraph_getBooksReadByReader_finite_invalid():
    g = SparseScoredBipartiteGraph(numReaders=5)
    g.readBook(3, 3, 1.5)
    assert_raises(IndexError, g.getBooksReadByReader, 5)
def test_SparseScoredBipartiteGraph_getBooksReadByReader_finite_empty():
    g = SparseScoredBipartiteGraph(numReaders=6)
    g.readBook(3, 3, 1.5)
    assert_equal(set(), g.getBooksReadByReader(5))
def test_SparseScoredBipartiteGraph_getBooksReadByReader_finite_nonempty():
    g = SparseScoredBipartiteGraph(numReaders=6, numBooks=10)
    g.readBook(3, 3, 1.5)
    g.readBook(3, 4, 1.5)
    assert_equal(set([3,4]), g.getBooksReadByReader(3))

def test_SparseScoredBipartiteGraph_getReadersOfBook_infinite_empty():
    g = SparseScoredBipartiteGraph()
    g.readBook(3, 3, 1.5)
    assert_equal(set(), g.getReadersOfBook(5))
def test_SparseScoredBipartiteGraph_getReadersOfBook_infinite_nonempty():
    g = SparseScoredBipartiteGraph()
    g.readBook(3, 1, 1.5)
    g.readBook(1, 1, 1.5)
    g.readBook(1, 2, 1.5)
    assert_equal(set([1,3]), g.getReadersOfBook(1))
def test_SparseScoredBipartiteGraph_getReadersOfBook_finite_invalid():
    g = SparseScoredBipartiteGraph(numBooks=5)
    assert_raises(IndexError, g.getReadersOfBook, 5)
def test_SparseScoredBipartiteGraph_getReadersOfBook_finite_empty():
    g = SparseScoredBipartiteGraph(numBooks=6)
    g.readBook(3, 3, 1.5)
    assert_equal(set(), g.getReadersOfBook(5))
def test_SparseScoredBipartiteGraph_getReadersOfBook_finite_nonempty():
    g = SparseScoredBipartiteGraph(numReaders=6, numBooks=10)
    g.readBook(3, 3, 1.5)
    g.readBook(4, 3, 1.5)
    g.readBook(4, 2, 1.5)
    assert_equal(set([3,4]), g.getReadersOfBook(3))

def test_SparseScoredBipartiteGraph_readBook_finite_valid():
    g = SparseScoredBipartiteGraph(numReaders=6, numBooks=10)
    assert not g.isReaderOfBook(3,5)
    g.readBook(3, 5, 1.5)
    assert g.isReaderOfBook(3,5)
    assert not g.isReaderOfBook(3,0)

def test_SparseScoredBipartiteGraph_readBook_finite_invalid():
    g = SparseScoredBipartiteGraph(numReaders=6, numBooks=10)
    assert_raises(IndexError, g.readBook, 6, 5, 1.5)
    assert_raises(IndexError, g.readBook, 3, 10, 1.5)

def test_SparseScoredBipartiteGraph_readBook_infinite():
    g = SparseScoredBipartiteGraph()
    assert not g.isReaderOfBook(1003,1005)
    g.readBook(1003, 1005, 1.5)
    assert g.isReaderOfBook(1003,1005)
    assert not g.isReaderOfBook(1003,1000)

def test_SparseScoredBipartiteGraph_isReaderOfBook_finite_invalid():
    g = SparseScoredBipartiteGraph(numReaders=6, numBooks=10)
    assert_raises(IndexError, g.isReaderOfBook, 6, 5)

def test_SparseScoredBipartiteGraph_isReaderOfBook_infinite_valid():
    g = SparseScoredBipartiteGraph()
    assert not g.isReaderOfBook(6, 5)
    
def test_SparseScoredBipartiteGraph_getReadingScore_infinite():
    g = SparseScoredBipartiteGraph()
    assert_almost_equal(0.0, g.getReadingScore(3,5))
    g.readBook(3, 5, 1.5)
    assert_almost_equal(1.5, g.getReadingScore(3,5))
    assert_almost_equal(0.0, g.getReadingScore(5,10))
def test_SparseScoredBipartiteGraph_getReadingScore_finite():
    g = SparseScoredBipartiteGraph(numReaders=6, numBooks=10)
    assert_raises(IndexError, g.getReadingScore, 6, 5)
    assert_raises(IndexError, g.getReadingScore, 5, 10)
    assert_almost_equal(0.0, g.getReadingScore(3,0))

