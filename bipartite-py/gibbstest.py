#!/usr/bin/env python

# TODO: rename, or even better, make into an automated module test
#                (nose assumes every module that contains "test" in its name 
#                to be a nose module test, so these should be moved to the test/ dir,
#                and contain only test functions that are evaluated automatically via asserts)

"""gibbstest.py: Applies Gibbs Sampler to synthetic data (generated via the generative algo)
        (according to [Caron 2012, Sec. 2.5 & 2.4])

    Outputs a comparison of true and sampled scores
"""

__author__ = "Matthias Sperber"

def usage():
    print """usage: gibbstest.py [options]
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
import source.gibbs as gibbs
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
        hyperParameters = expr.HyperParameters(alpha, sigma, tau)
        scores, sparseMatrix = gen.generateBipartiteGraph(hyperParameters, gammas )
   
    
                    
        scoresOutput = "artificial graph edges:\n"
        for line in sparseMatrix:
            scoresOutput += " ".join([str(i) for i in sorted(line)]) + "\n"
        sys.stderr.write(scoresOutput)
        scoresOutput = "\nartificial graph scores:\n"
        for line in scores:
            scoresOutput += str(line) + "\n"
        sys.stderr.write(scoresOutput)
        

        gParameters=expr.GraphParameters.deduceFromSparseGraph(sparseMatrix)
        numGibbsIterations = 10000
        us = gibbs.gibbsSampler(hyperParameters, gParameters, gammas, sparseMatrix,
                                numIterations = numGibbsIterations)  
            
#        plt.hist([us[s][0][0] for s in range(len(us))])
#        plt.savefig("test.png")

        scoresOutput = "\nestimated graph scores after %s iterations:\n" % numGibbsIterations
        for line in us[-1]:
            scoresOutput += str(line) + "\n"
        sys.stderr.write(scoresOutput)

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
    
