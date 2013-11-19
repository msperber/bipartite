'''
Created on Nov 11, 2013

@author: Matthias Sperber
'''

import math

class Parameters(object):
    def __init__(self, alpha, sigma, tau):
        assert 0 <= sigma < 1.0
        self.alpha = alpha
        self.sigma = sigma
        self.tau = tau

def lambdaFunction(w, parameters):
    assert w!=0.0
    alpha = parameters.alpha
    sigma = parameters.sigma
    tau = parameters.tau
    return (alpha/math.gamma(1.0-sigma)) * (w**(-sigma-1.0)) * (math.e ** (-w*tau)) # checked sv

def psiFunction(t, parameters):
    alpha = parameters.alpha
    sigma = parameters.sigma
    tau = parameters.tau
    if -0.000001<sigma<0.000001: # careful: division by zero
        return alpha * math.log(1.0 + t / tau) # checked sv
    else:
        return (alpha/sigma) * ((t + tau)**sigma - tau**sigma) # checked sv

def psiTildeFunction(t, b, parameters):
    # sv: For GGP it follows from Equations (3) (4) and the density of the GGP that 
    return psiFunction(t, Parameters(parameters.alpha, parameters.sigma, parameters.tau + b))

def kappaFunction(n, z, parameters):
    alpha = parameters.alpha
    sigma = parameters.sigma
    tau = parameters.tau
    return (alpha / (z+tau)**(n-sigma)) * (math.gamma(n-sigma)/math.gamma(1.0-sigma)) # checked sv
