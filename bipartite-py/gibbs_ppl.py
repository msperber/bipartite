#!/usr/bin/env python

"""
Description

Usage:
  gibbs_ppl.py [options] <corpusFile>

Options:
    --numDocs n     limit number of documents to be loaded
    --updateHyperParams
    --pplEachIteration
    --alpha x           alpha Hyperparameter [default: 5.0]
    --tau x             tau Hyperparameter [default: 1.0]
    --sigma x           sigma Hyperparameter [default: 0.0]
    --alphaF x          alphaF alphaF [default: 1.0]
    --alphaTheta x      alphaTheta alphaF [default: 1.0]
    --numIterations n   number of iterations [default: 100]
    --vocabSize n       max vocab size [default: 100]
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
import time

import source.topics.infer_topics as infer_topics
import source.document_data as document_data
from source.topics.infer_topics_hyperparam import HyperParameters

def main(argv=None):
    arguments = docopt.docopt(__doc__, options_first=True, argv=argv)
    corpusFileName = arguments['<corpusFile>']
    maxNumDocs = None
    if '--numDocs' in arguments: maxNumDocs = int(arguments['--numDocs'])
    updateHyperParams = arguments.get('--updateHyperParams', False)
    pplEachIteration = arguments.get('--pplEachIteration', False)
    alpha = float(arguments.get('--alpha'))
    sigma = float(arguments.get('--sigma'))
    tau = float(arguments.get('--tau'))
    alphaF = float(arguments.get('--alphaF'))
    alphaTheta = float(arguments.get('--alphaTheta'))
    numIterations = int(arguments.get('--numIterations'))
    maxVocabSize = int(arguments.get('--vocabSize'))
    
    ###########################
    ## MAIN PROGRAM ###########
    ###########################
    
    startTime = time.time()
    
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
    trainCorpus, testCorpus = textCorpus.split()
    print "done:"
    print "loaded", len(textCorpus), "documents", "with a total of", textCorpus.getTotalNumWords(), "words (using half for testing)"
    print "vocab size:", textCorpus.getVocabSize()
    print ""
    
    hyperParams = HyperParameters(alpha=alpha, sigma=sigma, tau=tau, alphaTheta=alphaTheta, 
                                          alphaF=alphaF, aGamma=1.0, bGamma=1.0)
    print "(INITIAL) HYPER PARAMETERS:", hyperParams
    print ""
    
    print "NUM ITERATIONS:",numIterations 
    
    numInitialTopics=10
    print "NUM INITIAL TOPICS:", numInitialTopics
    
    infer_topics.inferTopicsCollapsedGibbs(textCorpus=trainCorpus, 
                                           hyperParameters=hyperParams, 
                                           numIterations=numIterations, 
                                           numInitialTopics=numInitialTopics,
                                           updateHyperparameters=updateHyperParams,
                                           verbose=True,
                                           estimatePerplexityForSplitCorpus=testCorpus,
                                           pplEachIteration=pplEachIteration)
    
    elapsedTime = (time.time() - startTime)
    print "elapsed time:", elapsedTime, "seconds"

if __name__ == "__main__":
    sys.exit(main())
