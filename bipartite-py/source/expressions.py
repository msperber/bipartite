'''
Created on Nov 11, 2013

@author: Matthias Sperber
'''

import math
import source.prob as prob

def lambdaFunction(w, alpha, sigma, tau):
    assert w!=0.0
    return (alpha/math.gamma(1.0-sigma)) * (w**(-sigma-1.0)) * (math.e ** (-w*tau)) # checked sv

def psiFunction(t, alpha, sigma, tau):
    if -0.000001<sigma<0.000001: # careful: division by zero
        return alpha * math.log(1.0 + t / tau) # checked sv
    else:
        return (alpha/sigma) * ((t + tau)**sigma - tau**sigma) # checked sv

def psiTildeFunction(t, b, alpha, sigma, tau):
    # sv: For GGP it follows from Equations (3) (4) and the density of the GGP that 
    return psiFunction(t, alpha, sigma, tau + b)

def kappaFunction(n, z, alpha, sigma, tau):
    return (alpha / (z+tau)**(n-sigma)) * (math.gamma(n-sigma)/math.gamma(1.0-sigma)) # checked sv
