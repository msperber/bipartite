'''
Created on Nov 12, 2013

@author: Matthias Sperber
'''

import source.expressions as expr
import source.prob as prob
from nose.tools import assert_almost_equal
import math

def test_lambdaFunction_boundaries_alpha():
    assert_almost_equal(0.0, expr.lambdaFunction(0.1, alpha=0.0, sigma=0.9, tau=1.0))
def test_lambdaFunction_boundaries_sigma():
    assert_almost_equal(1.0, expr.lambdaFunction(1.0, alpha=1.0, sigma=0.0, tau=0.0))

def test_psiFunction_boundaries_sigma():
    assert_almost_equal(5.0, expr.psiFunction(math.e - 1.0, alpha=5.0, sigma=0.0, tau=1.0))
def test_psiFunction_boundaries_sigma2():
    assert_almost_equal(0.0, expr.psiFunction(0.0, alpha=5.0, sigma=0.5, tau=3.0))
def test_psiFunction_boundaries_tau():
    assert_almost_equal(1.0, expr.psiFunction(1.0**(-0.9), alpha=0.9, sigma=0.9, tau=0.0))
    
def test_psiTildeFunction_check1():
    assert_almost_equal(expr.psiFunction(1.0**(-0.9), alpha=0.9, sigma=0.9, tau=0.0),
                        expr.psiTildeFunction(1.0**(-0.9), 0.0, alpha=0.9, sigma=0.9, tau=0.0))
def test_psiTildeFunction_check2():
    assert_almost_equal(expr.psiFunction(1.0**(-0.9), alpha=0.9, sigma=0.9, tau=0.2),
                        expr.psiTildeFunction(1.0**(-0.9), 0.2, alpha=0.9, sigma=0.9, tau=0.0))
    
def test_kappaFunction_boundaries_sigma():
    assert_almost_equal(math.gamma(2.0)/4.0, expr.kappaFunction(2.0, 1.0, alpha=1.0, sigma=0.0, tau=1.0))
            
    