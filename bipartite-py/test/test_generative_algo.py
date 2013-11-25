'''
Created on Nov 12, 2013

@author: Matthias Sperber
'''

from nose.tools import assert_almost_equals

import source.generative_algo as gen
import source.expressions as expr
from source.prob import sampleFrom15

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
                                    simulationParameters=\
                                            expr.HyperParameters(alpha=0.1, sigma=0.0, tau=0.5),
                                    poisson=lambda x,y: 2,
                                    sampleFrom15=lambda a,b,c,d: 0.3)
    assert numBooks == initialNumBooks + 2
    assert bookScores[2] == 0.3
    assert bookScores[3] == 0.3
    print numBooks
    print bookScores

def test_generateBipartiteGraph_condition1():
    # TODO: implement
    assert False

