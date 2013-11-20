#!/usr/bin/env python

"""plot_bipartite.py: Plots a bipartite graph, currently by simply visualizing its adjacency matrix

    Input is a sparse matrix, where the rows are the readers, and each row contains the 
    (whitespace-seperated) (zero-base) IDs of all the books read by that reader
    (output of generate.py can be used)
    
    expected sparse matrix format:
    0 1 2 3
    0 3 4
    1 5 6
    ...
"""

__author__ = "Matthias Sperber"
__date__   = "Nov 11, 2013"

def usage():
    print """usage: generate.py [options] sparse-matrix-file
                    generate.py [options] < sparse-matrix
    -h --help: print this Help message
    -o --output-file f: write plot to file (default: display on screen)
"""

import getopt
import sys
from numpy import sum, array
import matplotlib.pyplot as plt

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
            optlist, args = getopt.getopt(argv[1:], 'ho:', ['help', 'output-file='])
        except getopt.GetoptError, msg:
            raise Usage(msg)
        outputFileName = None
        for o, a in optlist:
            if o in ["-h", "--help"]:
                usage()
                exit(2)
            if o in ["-o", "--output-file"]:
                outputFileName = a
        EXPECTED_NUM_PARAMS = [0,1]
        if len(args) not in EXPECTED_NUM_PARAMS:
            raise Usage("must contain %s non-optional parameter(s)" % (EXPECTED_NUM_PARAMS))
        dataLines = []
        if len(args)==1:
            dataLines = open(args[0]).readlines()
        else:
            dataLines = sys.stdin.readlines()
        sparseMatrix = [[int(s) for s in line.split()] for line in dataLines]
#       sparseMatrix = [[0, 1, 2, 3, 4, 5], [1, 3, 4, 6], [0, 1, 2, 3, 4, 7, 8], [0, 1, 2, 3, 4, 5, 6, 9]]

        ###########################
        ## MAIN PROGRAM ###########
        ###########################
        maxBookId = None
        for row in sparseMatrix:
            if len(row)>0:
                maxBookId = max(maxBookId, max(row))
        fullMatrix = []
        for row in sparseMatrix:
            curFullRow = []
            for j in range(maxBookId+1):
                if j in row:
                    curFullRow.append(0)
                else:
                    curFullRow.append(1)
            fullMatrix.append(curFullRow)
        
        data = array(fullMatrix)
        
        
        plt.imshow(data, cmap='Greys',  interpolation='nearest')
        plt.yticks(range(len(fullMatrix)),range(len(fullMatrix)))
        plt.ylabel('Readers')
        plt.xlabel('Books')
        if outputFileName is None:
            plt.show()
        else:
            plt.savefig(outputFileName)
        ###########################
        ###########################

    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use --help"
        return 2


if __name__ == "__main__":
    sys.exit(main())
    
