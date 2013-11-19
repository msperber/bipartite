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

        ###########################
        ## MAIN PROGRAM ###########
        ###########################
        simulationParameters = expr.Parameters(alpha, sigma, tau)
        print gen.generateBipartiteGraph(simulationParameters, [gamma] ** numReaders)
        
        ###########################
        ###########################

    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use --help"
        return 2


if __name__ == "__main__":
    sys.exit(main())
    
