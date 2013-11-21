from math import *
import random


def A(u,alpha):
    nom= sin(alpha*u)**alpha *( sin((1-alpha)*u)**(1-alpha) )
    return (nom/sin(u))**(1/(1-alpha))
    
def B(x,alpha):
    if abs(x)<1.0E-6:
        return alpha**(-alpha)*(1-alpha)**(-(1-alpha))
    return sin(x)/(sin(alpha*x)**alpha*(sin(1-alpha)*x)**(1-alpha))

	

def sampleGStarStar(lam,alpha):
    '''
		Sampling from g** in order to perform double rejection see [Devroye 2009, p.17]
    '''
    gamma=lam**alpha *alpha*(1-alpha)
    xi= ((2.0+sqrt(pi/2.0))*sqrt(2.0*gamma)+1.0)/pi
    psi=exp(-gamma*pi**2.0/8.0)/pi*(2.0+sqrt(pi/2.0))*sqrt(gamma*pi)
    w1= xi*sqrt(pi/(2.0*gamma))
    w2= 2.0*psi*sqrt(pi)
    w3= xi*psi
    V=random.random()
    W=random.random()
    if gamma>= 1:
        if V<= w1/(w1+w2):
	   return abs(random.gauss(0,1/sqrt(gamma)))
	else:
	   return pi * (1-W**2)
    else:
        if V<= w3/(w2+w3):
	   return pi *W
	else:
            return pi * (1-W**2)
            
            
def sampleExpTStable(lam,alpha):
    gamma=lam**alpha *alpha*(1-alpha)
    
    xi= ((2.0+sqrt(pi/2.0))*sqrt(2.0*gamma)+1.0)/pi
    psi=exp(-gamma*pi**2.0/8.0)/pi*(2.0+sqrt(pi/2.0))*sqrt(gamma*pi)
    w1= xi*sqrt(pi/(2.0*gamma))
    w2= 2.0*psi*sqrt(pi)
    w3= xi*psi
    b=(1-alpha)/alpha
    
    while True:
        while True:
            U=sampleGStarStar(lam,alpha)
            W=random.random()
            zeta =sqrt( B(U,alpha)/B(0,alpha))
            phi= (sqrt(gamma)+alpha*zeta)**(1.0/alpha)
            z=phi/( phi-sqrt(gamma)**(1.0/alpha) )
            #calculate big Bracket in definiton of rho
            rhoBracket=0
            if U<pi:
                rhoBracket=rhoBracket+psi/sqrt(pi-U)
                if gamma< 1:
                    rhoBracket=rhoBracket+xi
                else: 
                    rhoBracket= rhoBracket+ xi *exp(-gamma*U**2.0/2.0)
                rho= pi * exp(-lam**alpha*(1-zeta**(-2)))*rhoBracket/((1+sqrt(pi/2.0))*sqrt(gamma)/zeta+z)
                if W*rho<1:
                    break
        a=A(U,alpha)
        m=(b *lam/a)**alpha
        delta = sqrt(m*alpha/a)
        a1=delta*sqrt(pi/2.0)
        a2=delta
        a3=z/a
        s=a1+a2+a3
        Vp=random.random()
        X=0
        Np=0
        Ep=0
        
        if Vp < a1/s:
            Np=random.gauss(0,1)
            X=m-delta*abs(Np)
        elif Vp< a2/s:
            X= m+delta*random.random()
        else:
            Ep=-log(random.random())
            X= m+delta+Ep*a3
        
        if X>=0:
            E=-log(random.random())
            LHS= a*(X-m)+lam*(X**(-b)-m**(-b))
            if X<m:
                LHS=LHS-Np**2.0/2.0
            elif X>m+delta:
                LHS=LHS-Ep
            if LHS<=E:
                break
    return 1/X**b
            
        
for i in range(1000):
    sampleExpTStable(2,0.4)        
        
        
                
                 
            
            
			
    


