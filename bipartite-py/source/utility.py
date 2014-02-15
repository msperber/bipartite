'''
Created on Dec 25, 2013

@author: Matthias Sperber
'''

## very flexible version available at:
## taken from http://code.activestate.com/recipes/577124-approximately-equal/
## 
## however, switched to this simple version that's twice as fast:

def approx_equal(x, y, prec=1e-6):
    return abs(x-y) <= prec
