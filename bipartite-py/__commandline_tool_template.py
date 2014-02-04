#!/usr/bin/env python

"""
Description

Usage:
  template.py [options]

Options:
    (as always: -h for help)
"""

__author__ = "AUTHOR_MARKER"
__date__   = "DATE_MARKER"


import docopt
import sys
import operator


def main(argv=None):
	arguments = docopt.docopt(__doc__, options_first=True, argv=argv)

	###########################
	## MAIN PROGRAM ###########
	###########################


if __name__ == "__main__":
	sys.exit(main())
