'''
Created on Nov 12, 2013

@author: Matthias Sperber
'''

from nose.tools import assert_almost_equals

import source.generative_algo as gen
import source.prob as prob

def dummyPoisson(x,y): return 10
def dummyDistr15(a,b,c,d): return 0.5

def test_selectBooksForFirstReader():
    # using dummy functions: poisson distribution always returns 10, distr. (15) always 0.5
    numBooks, bookScores = gen.selectBooksForFirstReader(
                                                     hyperParameters=
                                                            prob.HyperParameters(1.0, 0.9, 1.0,
                                                                          gammas=[0]),
                                                     poisson=dummyPoisson,
                                                     sampleFrom15=dummyDistr15
                                                     ) 
    assert numBooks == 10
    assert len(bookScores) == 10
    print bookScores
    for score in bookScores.values(): assert_almost_equals(0.5, score)

def test_selectBooksForIthReader_condition1():
    initialNumBooks = 2
    numBooks, bookScores = gen.selectBooksForIthReader(
                                    numBooks=initialNumBooks, 
                                    prevBookScoreList=[{0:1.1, 1:1.1}],
                                    hyperParameters=\
                                            prob.HyperParameters(alpha=0.1, sigma=0.0, tau=0.5,
                                                                 gammas=[0.1, 0.1]),
                                    poisson=lambda x,y: 2,
                                    sampleFrom15=lambda a,b,c,d: 0.3)
    assert numBooks == initialNumBooks + 2
    assert bookScores[2] == 0.3
    assert bookScores[3] == 0.3
    print numBooks
    print bookScores

def test_generateBipartiteGraph_deterministicNumBooks():
    numReaders = 10
    gammas=[2]*numReaders
    hyperParameters = prob.HyperParameters(alpha=5.0, sigma=0.0, tau=1.0, gammas=gammas)
    bGraph = gen.generateBipartiteGraph(hyperParameters=hyperParameters, 
                                        poisson=lambda x,y: 1)
    for i in range(numReaders):
        # wp1, every reader is picking exactly one new book:
        assert bGraph.isReaderOfBook(i, i)

def test_generateBipartiteGraph_deterministicScores():
    numReaders = 10
    gammas=[2]*numReaders
    hyperParameters = prob.HyperParameters(alpha=5.0, sigma=0.0, tau=1.0, gammas=gammas)
    fixedScore=1.5
    bGraph = gen.generateBipartiteGraph(hyperParameters, 
                                        sampleFrom15=lambda a,b,c,d: fixedScore)
    for reader in range(numReaders):
        for book in bGraph.getBooksReadByReader(reader):
            assert_almost_equals(fixedScore, bGraph.getReadingScore(reader, book))
