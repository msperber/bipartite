#!/usr/bin/env python

"""plot_bipartite.py: Plots a bipartite graph, currently by simply visualizing its adjacency matrix

    Input is a sparse matrix, where the rows are the readers, and each row contains the 
    (whitespace-seperated) (zero-base) IDs of all the books read by that reader
    (output of generate.py can be used)
"""

__author__ = "Matthias Sperber"
__date__   = "Nov 11, 2013"

def usage():
    print """usage: generate.py [options] sparse-input
"""

import getopt
import sys
from numpy import sum, array
import pylab as plt

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
            optlist, args = getopt.getopt(argv[1:], 'h', ['help'])
        except getopt.GetoptError, msg:
            raise Usage(msg)
        for o, a in optlist:
            if o in ["-h", "--help"]:
                usage()
                exit(2)
        EXPECTED_NUM_PARAMS = 0 # TODO
        if len(args)!=EXPECTED_NUM_PARAMS:
            raise Usage("must contain %s non-optional parameter(s)" % (EXPECTED_NUM_PARAMS))
        inputFile = args[0]
        ###########################
        ## MAIN PROGRAM ###########
        ###########################
        sparseMatrix = [[int(s) for s in line.split()] for line in open(inputFile).readlines()]
#        sparseMatrix = [[0, 1, 2, 3, 4, 5], [1, 3, 4, 6], [0, 1, 2, 3, 4, 7, 8], [0, 1, 2, 3, 4, 5, 6, 9]]
        maxBookId = 0
        for row in sparseMatrix:
            maxBookId = max(maxBookId, max(row))
        fullMatrix = []
        for row in sparseMatrix:
            curFullRow = []
            for j in range(maxBookId+1):
                if j in row:
                    curFullRow.append(1)
                else:
                    curFullRow.append(0)
            fullMatrix.append(curFullRow)
        
        data = array(fullMatrix)
        
        
        plt.imshow(data, cmap='Greys',  interpolation='nearest')
        plt.yticks(range(len(fullMatrix)),range(len(fullMatrix)))
        plt.ylabel('Readers')
        plt.xlabel('Books')
        plt.show()
        ###########################
        ###########################

    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use --help"
        return 2


if __name__ == "__main__":
    sys.exit(main())
    
