'''
Created on Nov 30, 2013

@author: Matthias Sperber
'''

from scipy.sparse import *
from scipy.sparse.dok import dok_matrix
import numpy

class GraphParameters(object):
    def __init__(self,n,K,m,Ks):
        assert n >=0
        assert K >= 0
        self.n=n    # number of readers
        self.K=K    # number of books
        self.m=m    # number of times each book was read
        self.Ks=Ks # number of books each reader has read
        
    @staticmethod
    def deduceFromSparseGraph(sparseMatrix):
        n=len(sparseMatrix)
        # calculate K (num books) from matrix
        Ks=[]
        K=0
        for i in range(n):
            Ks.append(len(sparseMatrix[i]))
            if len(sparseMatrix[i])>0:
                K=max([K,max(sparseMatrix[i])])
        K+=1        
        # calculate m (num times each book was read)
        m=[0]*K
        for reader in sparseMatrix:
            for book in reader:
                m[book]+=1
        return GraphParameters(n,K,m,Ks)

class AbstractBipartiteGraph(object):
    """
    bipartite graph representation. books and readers are represented via 0-based indices.
    """
    def __init__(self, datatype, numReaders=None, numBooks=None):
        """
        if numReaders is specified, assume a finite set of readers; 
                                        no larger readerIndices can be added
        if it is omitted, assume an infinite set of readers
        """
        initNumReaders = numReaders if numReaders is not None else 0
        initNumBooks = numBooks if numBooks is not None else 0
        self.graph = dok_matrix((initNumReaders, initNumBooks), dtype=datatype)
        self.numReaders, self.numBooks = numReaders, numBooks
        self.activeReaders, self.activeBooks = set(), set()
    def getActiveReaders(self):
        """
        returns the set of active readers
        """
        return self.activeReaders
    def getNumActiveReaders(self):
        """
        returns the number of active readers
        """
        return len(self.activeReaders)
    def getActiveBooks(self):
        """
        returns the set of active books
        """
        return self.activeBooks
    def getNumActiveBooks(self):
        """
        returns the number of active books
        """
        return len(self.activeBooks)
    def getMaxNumReaders(self):
        """
        number of readers in graph
        """
        if self.numReaders is None: return float("inf")
        else: return self.numReaders
    def getMaxNumBooks(self):
        """
        number of books in graph
        """
        if self.numBooks is None: return float("inf")
        else: return self.numBooks
    def getBooksReadByReader(self, readerIndex):
        """
        return list of indices of all books read by given reader 
        """
        if self.numReaders is None and readerIndex>=self.graph.shape[0]:
            return set()
        else:
            return set(numpy.nonzero(self.graph[readerIndex, : ])[1])
    def getReadersOfBook(self, bookIndex):
        """
        return list of indices of all readers that read a given book
        """
        if self.numBooks is None and bookIndex>=self.graph.shape[1]:
            return set()
        else:
            return set(numpy.nonzero(self.graph[ : , bookIndex])[0])
    def readBook(self, readerIndex, bookIndex, score):
        if (self.numReaders is not None and readerIndex>=self.numReaders) \
                or (self.numBooks is not None and bookIndex>=self.numBooks):
            raise IndexError()
        self.activeReaders.add(readerIndex)
        self.activeBooks.add(bookIndex)
        if readerIndex >= self.graph.shape[0] or bookIndex >= self.graph.shape[1]:
            newNumReaders = max(readerIndex+1, self.graph.shape[0])
            newNumBooks = max(bookIndex+1, self.graph.shape[1])
            self.graph.resize((newNumReaders, newNumBooks))
        self.graph[readerIndex, bookIndex] = score
    
class SparseBinaryBipartiteGraph(AbstractBipartiteGraph):
    def __init__(self, numReaders=None, numBooks=None):
        super(SparseBinaryBipartiteGraph, self).__init__(datatype=numpy.int_,
                                                         numReaders=numReaders,
                                                         numBooks=numBooks)
    def readBook(self, readerIndex, bookIndex):
        super(SparseBinaryBipartiteGraph, self).readBook(readerIndex, bookIndex, 1)
    def isReaderOfBook(self, readerIndex, bookIndex):
        if (self.numReaders is None and readerIndex>=self.graph.shape[0]) \
                or (self.numBooks is None and bookIndex>=self.graph.shape[1]):
            return False
        else:
            return self.graph[readerIndex, bookIndex]==1

class SparseScoredBipartiteGraph(AbstractBipartiteGraph):
    def __init__(self, numReaders=None, numBooks=None):
        super(SparseScoredBipartiteGraph, self).__init__(datatype=numpy.float64,
                                                         numReaders=numReaders,
                                                         numBooks=numBooks)
    def readBook(self, readerIndex, bookIndex, score):
        super(SparseScoredBipartiteGraph, self).readBook(readerIndex, bookIndex, score)
    def isReaderOfBook(self, readerIndex, bookIndex):
        if (self.numReaders is None and readerIndex>=self.graph.shape[0]) \
                or (self.numBooks is None and bookIndex>=self.graph.shape[1]):
            return False
        else:
            return self.graph[readerIndex, bookIndex]>0
    def getReadingScore(self, readerIndex, bookIndex):
        if (self.numReaders is None and readerIndex>=self.graph.shape[0]) \
                or (self.numBooks is None and bookIndex>=self.graph.shape[1]):
            return 0.0
        else:
            return self.graph[readerIndex, bookIndex]
