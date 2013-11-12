'''
Created on Nov 12, 2013

@author: Matthias Sperber
'''

from nose.tools import assert_almost_equals

import source.generative_algo as gen
import source.expressions as expr

def dummyPoisson(x,y): return 10
def dummyDistr15(): return 0.5

def test_selectBooksForFirstReader():
    # using dummy functions: poisson distribution always returns 10, distr. (15) always 0.5
    numBooks, bookScores = gen.selectBooksForFirstReader(0,
                                                     expr.Parameters(1.0, 1.0, 1.0),
                                                     poissonFunction=dummyPoisson,
                                                     sampleFrom15Function=dummyDistr15
                                                     ) 
    assert numBooks == 10
    assert len(bookScores) == 10
    print bookScores
    for score in bookScores.values(): assert_almost_equals(0.5, score)

def test_selectBooksForIthReader_condition1():
    # TODO: implement
    assert False

def test_selectBooksForIthReader_condition2():
    # TODO: implement
    assert False

def test_generateBipartiteGraph_condition1():
    # TODO: implement
    assert False

