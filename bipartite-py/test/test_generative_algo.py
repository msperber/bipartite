'''
Created on Nov 12, 2013

@author: Matthias Sperber
'''

from nose.tools import assert_almost_equals

import source.generative_algo as gen
import source.expressions as expr

def dummyPoisson(x,y): return 10
def dummyDistr15(a,b,c,d): return 0.5

def test_selectBooksForFirstReader():
    # using dummy functions: poisson distribution always returns 10, distr. (15) always 0.5
    numBooks, bookScores = gen.selectBooksForFirstReader([0],
                                                     expr.HyperParameters(1.0, 0.9, 1.0),
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
                                    gammas=[0.1, 0.1], 
                                    numBooks=initialNumBooks, 
                                    prevBookScoreList=[{0:1.1, 1:1.1}],
                                    hyperParameters=\
                                            expr.HyperParameters(alpha=0.1, sigma=0.0, tau=0.5),
                                    poisson=lambda x,y: 2,
                                    sampleFrom15=lambda a,b,c,d: 0.3)
    assert numBooks == initialNumBooks + 2
    assert bookScores[2] == 0.3
    assert bookScores[3] == 0.3
    print numBooks
    print bookScores

def test_generateBipartiteGraph_deterministicNumBooks():
    hyperParameters = expr.HyperParameters(alpha=5.0, sigma=0.0, tau=1.0)
    gammas=[2]*10
    _, sparseMatrix = gen.generateBipartiteGraph(hyperParameters, 
                                                      gammas, poisson=lambda x,y: 1)
    for i in range(len(sparseMatrix)):
        # wp1, every reader is picking exactly one new book:
        assert i in sparseMatrix[i]

def test_generateBipartiteGraph_deterministicScores():
    hyperParameters = expr.HyperParameters(alpha=5.0, sigma=0.0, tau=1.0)
    gammas=[2]*10
    fixedScore=1.5
    scores, sparseMatrix = gen.generateBipartiteGraph(hyperParameters, 
                                                      gammas, sampleFrom15=lambda a,b,c,d: fixedScore)
    for reader in range(len(sparseMatrix)):
        for book in sparseMatrix[reader]:
            assert_almost_equals(fixedScore, scores[reader][book])
