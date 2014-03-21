# based on code by Francois Caron

import numpy as np
import sys

from math import *
def establernd(V0,alpha,tau,n):

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% Random variate generation for the exponentially tilted stable distribution 
#% with Laplace transform (in z)
#% exp(-V0 * ((z + tau)**alpha - tau**alpha))
#% Uses the algorithm proposed in (Devroye, 2009) 
#% with corrections pointed out in (Hofert, 2011)
#%
#% References:
#% Luc Devroye. Random variate generation for exponentially and polynomially
#% tilted stable distributions. ACM Transactions on Modeling and Computer
#% Simulation, vol. 19(4), 2009.
#%
#% Marius Hofert. Sampling exponentially tilted stable distributions. ACM Transactions on Modeling and Computer
#% Simulation, vol. 22(1), 2011.
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#% Franois Caron
#% INRIA Bordeaux Sud-Ouest
#% May 2012
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
#% Check parameters
    assert alpha >0 and alpha <1
    assert tau >= 0
    assert V0 >0 

    lamb = tau * V0**(1.0/alpha)
    lam_alpha = tau**alpha * V0 



    #% sigma, lam, as in (Devroye, 2009)
    gamma = lam_alpha * alpha * (1.0-alpha)

    xi = 1/pi *((2+sqrt(pi/2.0)) * sqrt(2.0*gamma) + 1.0)# Correction in Hofert
    psi = 1.0/pi * exp(-gamma * pi**2/8) * (2 + sqrt(pi/2.0)) * sqrt(gamma * pi)
    w1 = xi * sqrt(pi/2.0/gamma)
    w2 = 2.0 * psi * sqrt(pi)
    w3 = xi * pi
    b = (1.0-alpha)/alpha

    samples = np.zeros(n)
    for i in range(n):
        while 1:
            # generate U with density g*/G*
            while 1:
                #Generate U with density proportional to g**
                U = gen_U(w1, w2, w3, gamma)
    
                W = np.random.rand()
                try:
                    zeta = sqrt(ratio_B(U, alpha))
                except ValueError:
                    print "breakpoint"
                    raise ValueError
                z = 1.0/(1.0 - (1.0 + alpha*zeta/sqrt(gamma))**(-1.0/alpha))
#                 print "lam_alpha ", lam_alpha , " zeta ", zeta , "\n"
#                 print "T1 ",  pi * exp(-lam_alpha * (1.0-zeta**(-2.0))), "\n" 
#                 sys.stdout.flush()
                
                         
                T2= (xi * exp(-gamma*U**2/2.0) * (U>=0)*(gamma>=1.0) + psi/sqrt(pi-U)* (U>0)*(U<pi) + xi *(U>=0)*(U<=pi)*(gamma<1.0))                                                      
                logRho = log(pi)+(-lam_alpha * (1.0-zeta**(-2.0)))+log(T2)-log((1.0 + sqrt(pi/2.0)) *sqrt(gamma)/zeta + z)
                
                
                #rho = pi * exp(-lam_alpha * (1.0-zeta**(-2.0))) \
                #    * (xi * exp(-gamma*U**2/2.0) * (U>=0)*(gamma>=1.0) + psi/sqrt(pi-U)* (U>0)*(U<pi) + xi *(U>=0)*(U<=pi)*(gamma<1.0)) \
                #    /((1.0 + sqrt(pi/2.0)) *sqrt(gamma)/zeta + z)
                
                     
                if (U<pi and logRho<=-log(W)):
                    break
                
    
            # Generate X with density proportional to g(x, U)
            a = zolotarev(U, alpha)
            m = (b/a)**alpha * lam_alpha
            delta = sqrt(m*alpha/a)
            a1 = delta * sqrt(pi/2.0)
            a2 = delta
            a2 = a1 + delta # correction in Hofert
            a3 = z/a
            s = a1 + delta + a3 # correction in Hofert
            V_p = np.random.rand()    
            N_p = np.random.normal()
            E_p = np.random.exponential()
            if V_p<a1/s:
                X = m - delta*abs(N_p)
            elif V_p<a2/s:
                X = delta * np.random.rand() + m
            else:
                X = m + delta + a3 * E_p
    
            E = np.random.exponential()
    
            cond = (a*(X-m) + exp(1.0/alpha*log(lam_alpha)-b*log(m))*((m/X)**b - 1.0) - N_p**2/2.0 * (X<m) - E_p * (X>m+delta))
            if (X>=0) and (cond <=E):
                break
    
        samples[i] = exp( 1.0/alpha* log(V0) -b*log(X)) # more stable than V0**(1/alpha) * X**(-b)
        
    return samples


def gen_U(w1, w2, w3, gamma):
    V = np.random.rand()
    W_p = np.random.rand()
    if gamma>=1.0:
        if (V < w1/(w1+w2)):
            U = abs(np.random.normal()) /sqrt(gamma)
        else:
            U = pi * (1.0 - W_p**2)    
    else:
        if (V < w3/(w3 + w2)):
            U = pi * W_p
        else:
            U = pi * (1.0 - W_p**2)
    return U


def ratio_B(x, sigma):
    return sinc(x) / (sinc(sigma * x))**sigma / (sinc((1.0-sigma)*x))**(1.0-sigma)


def sinc(x):
    return sin(x)/x


def zolotarev(u, sigma):
# Zolotarev function, cf (Devroye, 2009)
   return  ((sin(sigma*u))**sigma * (sin((1.0-sigma)*u))**(1.0-sigma) / sin(u))**(1.0/(1.0-sigma))
