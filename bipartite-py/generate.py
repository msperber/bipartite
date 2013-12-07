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
    -g --gamma x: (uniform) gamma parameter > 0
    -G --Gammas l: comma-separated list of gamma parameters for each reader
    -a --alpha x: Alpha parameter
    -t --tau x: Tau parameter
    -s --sigma x: Sigma parameter
    -n --num-readers x: Fixed number or readers (int, >= 1)
    -A --A-param x: specify parameter "a" (shape parameter of distribution over gammas)
    -B --B-param x: specify parameter "b" (scale parameter of distribution over gammas)

    example: generate.py -g 2 -a 5 -t 1 -s 0.5 -n 10
"""

import getopt
import sys
import source.generative_algo as gen
import source.prob as prob

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
            optlist, args = getopt.getopt(argv[1:], 'hg:G:a:t:s:n:A:B:',
                                                     ['help', 'gamma=',
                                                      'alpha=', 'tau=',
                                                      'sigma=', 'num-readers=',
                                                      'Gammas=', 'A-param=',
                                                      'B-param='])
        except getopt.GetoptError, msg:
            raise Usage(msg)
        gamma, alpha, tau, sigma, numReaders = None, None, None, None, None
        gammas, aParam, bParam = None, None, None
        for o, a in optlist:
            if o in ["-h", "--help"]:
                usage()
                exit(2)
            if o in ["-g", "--gamma"]:
                gamma = float(a)
                assert a > 0
            if o in ["-G", "--Gammas"]:
                gammas = [float(g) for g in a.split(",")]
                for g in gammas:
                    assert g > 0
            if o in ["-a", "--alpha"]:
                alpha = float(a)
            if o in ["-t", "--tau"]:
                tau = float(a)
            if o in ["-s", "--sigma"]:
                sigma = float(a)
            if o in ["-n", "--num-readers"]:
                numReaders = int(a)
            if o in ["-A", "--A-param"]:
                aParam = float(a)
            if o in ["-B", "--B-param"]:
                bParam = float(a)
        EXPECTED_NUM_PARAMS = 0
        if len(args)!=EXPECTED_NUM_PARAMS:
            raise Usage("must contain %s non-optional parameter(s)" % (EXPECTED_NUM_PARAMS))
        gammaPriorSpecified = aParam is not None and bParam is not None and numReaders is not None
        uniformGammaSpecified = gamma is not None and numReaders is not None
        individualGammasSpecified = gammas is not None
        if len([x for x in [gammaPriorSpecified,uniformGammaSpecified,individualGammasSpecified] if x==True])!=1:
            raise Usage("must specify either gammas, or uniform gamma and numReaders, or a and b and numReaders")
        if gammas is not None and numReaders is not None and len(gammas)!=numReaders: 
            raise Usage("num gamma values and num readers not consistent")
        if None in [alpha, tau, sigma]:
            raise Usage("must specify alpha, tau, sigma")

        ###########################
        ## MAIN PROGRAM ###########
        ###########################
        if gammaPriorSpecified:
            hyperParameters = prob.HyperParameters(alpha=alpha, sigma=sigma, tau=tau,
                                                    a=aParam, b=bParam, numReaders=numReaders)
        elif uniformGammaSpecified:
            hyperParameters = prob.HyperParameters(alpha=alpha, sigma=sigma, tau=tau,
                                                    gammas=[gamma] * numReaders, numReaders=numReaders)
        else: # individualGammasSpecified
            hyperParameters = prob.HyperParameters(alpha=alpha, sigma=sigma, tau=tau,
                                                    gammas=gammas, numReaders=numReaders)
        hyperParametersWithGammas = prob.HyperParameters.sampleGammasIfNecessary(hyperParameters)
        bGraph = gen.generateBipartiteGraph(hyperParametersWithGammas)
        scoresOutput = "GAMMAS:" + str(hyperParametersWithGammas.gammas) + "\n"
        scoresOutput += "SCORES:\n"
        scoresOutput += bGraph.summarizeScores()
        scoresOutput += "GRAPH:\n"
        scoresOutput += bGraph.summarizeGraph()
        sys.stderr.write(scoresOutput)
        
        matrixOutput = ""
        for reader in range(numReaders):
            matrixOutput += " ".join([str(i) for i in sorted(bGraph.getBooksReadByReader(reader))]) + "\n"
        sys.stdout.write(matrixOutput)
                    
        ###########################
        ###########################

    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use --help"
        return 2


if __name__ == "__main__":
    sys.exit(main())
    
