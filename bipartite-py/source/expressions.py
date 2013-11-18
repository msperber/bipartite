'''
Created on Nov 11, 2013

@author: Matthias Sperber
'''

import math

class Parameters(object):
    def __init__(self, alpha, sigma, tao):
        assert 0 <= sigma < 1.0
        self.alpha = alpha
        self.sigma = sigma
        self.tao = tao

def lambdaFunction(w, parameters):
    assert w!=0.0
    alpha = parameters.alpha
    sigma = parameters.sigma
    tao = parameters.tao
    return (alpha/math.gamma(1.0-sigma)) * (w**(-sigma-1.0)) * (math.e ** (-w*tao))

def psiFunction(t, parameters):
    alpha = parameters.alpha
    sigma = parameters.sigma
    tao = parameters.tao
    if -0.000001<sigma<0.000001: # careful: division by zero
        return alpha * math.log(1.0 + t / tao)
    else:
        return (alpha/sigma) * ((t + tao)**sigma - tao**sigma)

def psiTildeFunction(t, b, parameters):
    # TODO: implement
    pass

def kappaFunction(n, z, parameters):
    alpha = parameters.alpha
    sigma = parameters.sigma
    tao = parameters.tao
    return (alpha / (z+tao)**(n-sigma)) * (math.gamma(n-sigma)/math.gamma(1.0-sigma))
