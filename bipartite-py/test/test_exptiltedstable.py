#Based on code by Francois Caron

from source.exptiltedstable import *
from scipy import integrate
from numpy import *
from pylab import *

def etstablepdf(x, V0, alpha, tau):

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# % pdf of the exponentially tilted stable distribution 
# % with Laplace transform (in z)
# % exp(-V0 * ((z + tau)**alpha - tau**sigma))
# % Can be expressed with Zolotarev's integral representation
# %
# % References:
# % Luc Devroye. Random variate generation for exponentially and polynomially
# % tilted stable distributions. ACM Transactions on Modeling and Computer
# % Simulation, vol. 19(4), 2009.
# %
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# % Franois Caron
# % INRIA Bordeaux Sud-Ouest
# % May 2012
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    assert alpha >0 and alpha <1.0
    assert tau >= 0
    assert V0 >0 



    lam = tau * V0**(1.0/alpha) #rescale - may be numerical issues
    lambda_alpha = tau**alpha * V0 # lambda**a

    x_alpha = x**alpha/V0
    x = x_alpha**(1.0/alpha)



    fun =lambda u: zolotarev(u, alpha)*exp(-zolotarev(u, alpha)/(x_alpha**(1.0/(1.0-alpha))) )
    integral = integrate.quad(fun, 0, pi)[0] # second component is error
    logpdf = lambda_alpha - lam * x \
    + log(alpha) - log(1.0-alpha) -1.0/(1.0-alpha)*log(x) - log(pi)\
    + log(integral)
    logpdf = logpdf - 1.0/alpha*log(V0)
    pdf = exp(logpdf)
    return np.array([pdf,logpdf])

def  zolotarev(u, sigma):
# Zolotarev function, cf (Devroye, 2009)
    return ((sin(sigma*u))**sigma * (sin((1.0-sigma)*u))**(1.0-sigma) / sin(u))**(1.0/(1.0-sigma))





# Check if empirical cdf is approx equal to true cdf

alpha = 1.0
tau = 1.0
sigma = 0.1
n = 10000
samples = establernd(alpha/sigma, sigma, tau, n)

pas = .01
x0 =  np.arange(pas,10,pas)
out = np.zeros(x0.size)
for i in range(x0.size):
    out[i] = etstablepdf(x0[i], alpha/sigma, sigma, tau)[0]

#plt.hist(samples, n, range, normed, weights, cumulative, bottom, histtype, align, orientation, rwidth, log, color, label, hold)

figure(0)
hist(samples, n, cumulative=True, color='green',normed=True)
plot(x0,pas*cumsum(out))
show()


# % Check computational time w.r.t. sigma
# alpha = 1
# tau = 1
# sigma_all = [.000001, .005, .01:.01:.09]
# n = 10000
# 
# for i=1:length(sigma_all)
#     tic
#     sigma = sigma_all(i)
#     samples = etstablernd(alpha/sigma, sigma, tau, n)
#     time_sig(i) = toc
# end
# 
# figure
# plot(sigma_all, time_sig)
# 

# Test if this is ok with gamma when sigma is small
# alpha = 1
# tau = 1
# sigma = 0.0001
# n = 1000
# samples = establernd(alpha/sigma, sigma, tau, n)
#  
# samples2 = np.random.gamma(alpha, 1/tau, n)
#  
# figure(2)
# subplot(2,1,1)
# hist(samples)
# subplot(2,1,2)
# hist(samples2)
# show()
#  
#[h, p] = kstest2(samples, samples2)
 
#  
# % Second test with gamma
# alpha = 3
# tau = 10
# sigma = 0.0001
# n = 1000
# samples = etstablernd(alpha/sigma, sigma, tau, n)
#  
# samples2 = gamrnd(alpha, 1/tau, n, 1)
#  
# figure
# subplot(2,1,1)
# hist(samples)
# subplot(2,1,2)
# hist(samples2)
#  
# [h, p] = kstest2(samples, samples2)
#  
#  
# % test with inverse gaussian (sigma = 0.5)
# % mu = 1
# % lambda = 2
#  
# % 
# % alpha = sqrt(lambda/2)
# % tau = sqrt(lambda/2/mu^2)
# alpha = 1
# tau = 2
# sigma = 0.5
# samples = etstablernd(alpha/sigma, sigma, tau, n)
#  
# lambda = 2*alpha^2
# mu = alpha/sqrt(tau)
# samples2 = igaussrnd(mu, lambda, n, 1)
#  
# figure
# subplot(2,1,1)
# hist(samples)
# subplot(2,1,2)
# hist(samples2)
# [h, p] = kstest2(samples, samples2)
#  
#  
#  
#  
# print etstablepdf(0.5, 1.0, 0.5, 1.0)
# print establernd(1.0,0.5,1.0,1000)
