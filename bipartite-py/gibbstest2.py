#!/usr/bin/env python

# TODO: rename, or even better, make into an automated module test
#                (nose assumes every module that contains "test" in its name 
#                to be a nose module test, so these should be moved to the test/ dir,
#                and contain only test functions that are evaluated automatically via asserts)

"""gibbstest2.py: Applies Gibbs Sampler to synthetic data (generated via the generative algo)
        (according to [Caron 2012, Sec. 2.5 & 2.4])

    Outputs a comparison of true and sampled scores
"""

__author__ = "Matthias Sperber"

def usage():
    print """usage: gibbstest2.py [options]
    -h --help: print this Help message
"""

import getopt
import sys
import source.generative_algo as gen
import source.prob as prob
import source.gibbs as gibbs
import source.graph as graph

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
            optlist, args = getopt.getopt(argv[1:], 'h', ['help'])
        except getopt.GetoptError, msg:
            raise Usage(msg)
        for o, a in optlist:
            if o in ["-h", "--help"]:
                usage()
                exit(2)
        EXPECTED_NUM_PARAMS = 0
        if len(args)!=EXPECTED_NUM_PARAMS:
            raise Usage("must contain %s non-optional parameter(s)" % (EXPECTED_NUM_PARAMS))

        ###########################
        ## MAIN PROGRAM ###########
        ###########################

        gamma=2.0
        alpha=2.0
        tau=1.0
        sigma=0.0
        numReaders=30
        ga=2
        gb=2 
        gammas=[gamma] * numReaders
        hyperParameters = prob.HyperParameters(alpha, sigma, tau, gammas=gammas, a=ga, b=gb)
        bGraph = gen.generateBipartiteGraph(hyperParameters)
        
                    
        scoresOutput = "artificial graph edges:\n"
        scoresOutput += bGraph.summarizeGraph()
        sys.stderr.write(scoresOutput)
        scoresOutput = "\nartificial graph scores:\n"
        scoresOutput += bGraph.summarizeScores()
        sys.stderr.write(scoresOutput)
        
        
        gParameters=graph.GraphParameters.deduceFromSparseGraph(bGraph)
        numGibbsIterations = 10000
        us = gibbs.gibbsSamplerPGammas(hyperParameters, gParameters, bGraph,
                                numGibbsIterations)  
                          
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
    
