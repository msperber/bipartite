#!/usr/bin/env python

"""SCRIPTNAME_MARKER.py: Description of script

"""

__author__ = "Matthias Sperber"
__date__   = "DATE_MARKER"

def usage():
	print """usage: SCRIPTNAME_MARKER.py [options] args
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
		EXPECTED_NUM_PARAMS = 0
		if len(args)!=EXPECTED_NUM_PARAMS:
			raise Usage("must contain %s non-optional parameter(s)" % (EXPECTED_NUM_PARAMS))

		###########################
		## MAIN PROGRAM ###########
		###########################
		
		
		###########################
		###########################

    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use --help"
        return 2


if __name__ == "__main__":
    sys.exit(main())
    
