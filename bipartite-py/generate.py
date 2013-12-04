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

    example: generate.py -g 2 -a 5 -t 1 -s 0.5 -n 10
"""

import getopt
import sys
import source.generative_algo as gen
import source.expressions as expr

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
        hyperParameters = expr.HyperParameters(alpha, sigma, tau)
        bGraph = gen.generateBipartiteGraph(hyperParameters, [gamma] * numReaders)
        scoresOutput = "SCORES:\n"
        scoresOutput += bGraph.summarizeGraph()
        scoresOutput += "GRAPH:\n"
        scoresOutput += bGraph.summarizeScores()
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
    
