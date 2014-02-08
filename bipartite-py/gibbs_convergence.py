#!/usr/bin/env python

"""
Description

Usage:
  gibbs_convergence.py [options] <corpusFile>

Options:
    --numDocs n     limit number of documents to be loaded
    --updateHyperParams
    (as always: -h for help)
"""
from source.prob import HyperParameters

__author__ = "Matthias Sperber"
__date__   = "Feb 4, 2014"


import docopt
import sys
import operator
import random
import numpy.random

import source.topics.infer_topics as infer_topics
import source.document_data as document_data
from source.topics.infer_topics_hyperparam import HyperParameters

def main(argv=None):
    arguments = docopt.docopt(__doc__, options_first=True, argv=argv)
    corpusFileName = arguments['<corpusFile>']
    maxNumDocs = None
    if '--numDocs' in arguments: maxNumDocs = int(arguments['--numDocs'])
    updateHyperParams = False
    if '--updateHyperParams' in arguments:
        updateHyperParams = True
    # TODO: make other parameters configurable
    
    ###########################
    ## MAIN PROGRAM ###########
    ###########################
    
    random.seed(13)
    numpy.random.seed(13)
    
    print "LOADING CORPUS", corpusFileName
    print "loading limited # docs:", maxNumDocs
    lowercase = True
    print "lowercasing:", lowercase
    minTokenLen=0
    print "removing tokens smaller than:", minTokenLen
    removeStopWords=True
    print "removing stopwords:", removeStopWords
    maxVocabSize=None
    print "limit vocab size:", maxVocabSize
    minNumTokens=10
    print "remove tokens with frequency <", minNumTokens
    print "loading.."
    textCorpus = document_data.DocumentCorpus.loadFromCorpusFile(\
                            corpusFile=corpusFileName, 
                            maxNumDocs=maxNumDocs, 
                            lowercase=lowercase, 
                            minTokenLen=minTokenLen, 
                            removeStopWords=removeStopWords, 
                            maxVocabSize=maxVocabSize, 
                            minNumTokens=minNumTokens)
    print "done:"
    print "loaded", len(textCorpus), "documents", "with a total of", textCorpus.getTotalNumWords(), "words"
    print "vocab size:", textCorpus.getVocabSize()
    print ""
    
    hyperParams = HyperParameters(alpha=5.0, sigma=0.0, tau=1.0, alphaTheta=1.0, 
                                          alphaF=1.0, aGamma=1.0, bGamma=1.0)
    print "(INITIAL) HYPER PARAMETERS:", hyperParams
    print ""
    
    numIterations=100
    print "NUM ITERATIONS:",numIterations 
    
    numInitialTopics=10
    print "NUM INITIAL TOPICS:", numInitialTopics
    
    infer_topics.inferTopicsCollapsedGibbs(textCorpus=textCorpus, 
                                           hyperParameters=hyperParams, 
                                           numIterations=numIterations, 
                                           numInitialTopics=numInitialTopics,
                                           updateHyperparameters=updateHyperParams,
                                           verbose=True)
    


if __name__ == "__main__":
    sys.exit(main())
