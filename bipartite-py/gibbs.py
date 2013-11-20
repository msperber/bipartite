#!/usr/bin/env python

"""generate.py: Generative algorithm, creates a power-law governed bipartite graph.
        (according to [Caron 2012, Sec. 2.4])

    Outputs a sparse matrix, where the rows are the readers, and each row contains the 
    (whitespace-seperated) numbers of all the books read by that reader
"""

__author__ = "Matthias Sperber"
__date__   = "Nov 11, 2013"

def usage():
    print """usage: generate.py [options]
    -h --help: print this Help message
    -g --gamma x: (Uniform) gamma parameter > 0
    -a --alpha x: Alpha parameter
    -t --tau x: Tau parameter
    -s --sigma x: Sigma parameter
    -n --num-readers x: Fixed number or readers (int, >= 1)
"""

import getopt
import sys
import source.generative_algo as gen
import source.expressions as expr
import source.prob as prob
import random
import matplotlib.pyplot as plt


class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg
class ModuleTest(Exception):
    def __init__(self, msg):
        self.msg = msg


def main(argv=None):
    if argv is None:
        argv = sys.argv
    try:
        try:
            optlist, args = getopt.getopt(argv[1:], 'hg:a:t:s:n:', ['help', 'gamma=',
                                                                    'alpha=', 'tau=',
                                                                    'sigma=', 'num-readers='])
        except getopt.GetoptError, msg:
            raise Usage(msg)
        gamma, alpha, tau, sigma, numReaders = None, None, None, None, None
        for o, a in optlist:
            if o in ["-h", "--help"]:
                usage()
                exit(2)
            if o in ["-g", "--gamma"]:
                gamma = float(a)
                assert a > 0
            if o in ["-a", "--alpha"]:
                alpha = float(a)
            if o in ["-t", "--tau"]:
                tau = float(a)
            if o in ["-s", "--sigma"]:
                sigma = float(a)
            if o in ["-n", "--num-readers"]:
                numReaders = int(a)
        EXPECTED_NUM_PARAMS = 0
        if len(args)!=EXPECTED_NUM_PARAMS:
            raise Usage("must contain %s non-optional parameter(s)" % (EXPECTED_NUM_PARAMS))
        if None in [gamma, alpha, tau, sigma, numReaders]:
            raise Usage("must specify gamma, alpha, tau, sigma, numReaders")

        ###########################
        ## MAIN PROGRAM ###########
        ###########################
        gammas=[gamma] * numReaders
        simulationParameters = expr.Parameters(alpha, sigma, tau)
        scores, sparseMatrix = gen.generateBipartiteGraph(simulationParameters, gammas )
   
    
                    
        scoresOutput = "SCORES:\n"
        for score in scores:
            scoresOutput += str(score) + "\n"
        scoresOutput += "GRAPH:\n"
        scoresOutput += str(sparseMatrix) + "\n"
        sys.stderr.write(scoresOutput)
        
        matrixOutput = ""
        for line in sparseMatrix:
            matrixOutput += " ".join([str(i) for i in sorted(line)]) + "\n"
        sys.stdout.write(matrixOutput)
        
            
        n=numReaders
        # calculate K from matrix
        K=0
        for row in sparseMatrix:
            if len(row)>0:
                K=max([K,max(row)])
        K+=1        
        # calculate m
        m=[0]*K
        for reader in sparseMatrix:
            for book in reader:
                m[book]+=1
        print "m:"
        print m 
        
        #init gibbs sampler 
        w= [1]*K
        
        us = []
        S=10000
        for s in range(S):
            # sample u given w
            u =[]
            for reader in sparseMatrix:
                u.append({})
            for i in range(n):
                for j in sparseMatrix[i]:
                    u[i][j]=prob.sampleTExp1(gammas[i]*w[j])
            # sample w given u
            for j in range(K):
                gammaSum= sum([u[i].get(j, 0.0) for i in range(n)])
                w[j] = random.gammavariate(m[j]-sigma,1/(tau+gammaSum))
            # save u for prediction
            us.append(u)  
            
        plt.hist([us[s][0][0] for s in range(S)])
        plt.savefig("test.png")
        print scores[0][0]
        print "true u00=%f\n" % scores[0][0] 
        
              
            
        
                
                
#          x ** (alpha - 1) * math.exp(-x / beta)
#pdf(x) =  --------------------------------------
#            math.gamma(alpha) * beta ** alpha
            
            
                    
            
        
        
        
        
                    
        ###########################
        ###########################

    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use --help"
        return 2


if __name__ == "__main__":
    sys.exit(main())
    
