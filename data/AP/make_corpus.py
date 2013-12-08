#!/usr/bin/env python

"""make_corpus.py: Removes xml tags (more precisely: all lines that start with "<")

"""

__author__ = "Matthias Sperber"
__date__   = "Dec 7, 2013"

def usage():
	print """usage: make_corpus.py [options] in.xml out.corpus
	-h --help: print this Help message
"""

import getopt
import sys

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
		EXPECTED_NUM_PARAMS = 2
		if len(args)!=EXPECTED_NUM_PARAMS:
			raise Usage("must contain %s non-optional parameter(s)" % (EXPECTED_NUM_PARAMS))

		inFileName = args[0]
		outFileName = args[1]
		
		###########################
		## MAIN PROGRAM ###########
		###########################
		
		outF = open(outFileName, "w")
		for line in open(inFileName):
			sentence = line.strip()
			if not sentence.startswith("<"):
				outF.write(sentence + "\n")	
		outF.close()
		
		###########################
		###########################

    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use --help"
        return 2


if __name__ == "__main__":
    sys.exit(main())
    
