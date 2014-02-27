#!/usr/bin/env python

"""
Tune alphaF and alphaTheta hyperparameters via Powell's search so as to minimize perplexity

Usage:
  tune_ppl_powell.py [options] <corpusFile>

Options:
    --numDocs n              limit number of documents to be loaded
    --updateHyperParams      hyper-parameter updates will be performed only if this flag is set
    --alpha x                alpha Hyperparameter [default: 5.0]
    --tau x                  tau Hyperparameter [default: 1.0]
    --sigma x                sigma Hyperparameter [default: 0.0]
    --alphaF x               initial alphaF hyperparameter [default: 1.0]
    --alphaTheta x           initial alphaTheta hyperparameter [default: 1.0]
    --numIterationsPerRun n  number of iterations per parameter combination[default: 50]
    --vocabSize n            max vocab size [default: 100]
    --alphaF_range s         string that denotes tuning range: min_max_nsteps [default: 0.1_5.0_10]
    --alphaTheta_range s     string that denotes tuning range: min_max_nsteps [default: 0.1_5.0_10]
    --numTuningIterations n  number of iterations in Powell's search [default: 10]
    --log f                  log file in which to print output (instead of stdout)
    (as always: -h for help)
"""
from source.prob import HyperParameters

__author__ = "Matthias Sperber"
__date__   = "Feb 18, 2014"


import docopt
import sys
import operator
import random
import numpy.random
import time
import copy

import source.topics.infer_topics as infer_topics
import source.document_data as document_data
from source.topics.infer_topics_hyperparam import HyperParameters


def parseChoices(rangeStr):
    minVal = float(rangeStr.split("_")[0])
    maxVal = float(rangeStr.split("_")[1])
    nSteps = int(rangeStr.split("_")[2])
    choices = [minVal] + [minVal + (maxVal-minVal)*(i+1)/(nSteps-1) for i in range(nSteps-1)]
    return choices
    
def seedRandom():
    random.seed(13)
    numpy.random.seed(13)

def customPrint(logF, *k):
    string = " ".join([str(i) for i in k])
    if logF is None:
        print string
    else:
        f = open(logF, "a")
        f.write(string + "\n")
        f.flush()
        f.close()
    
def main(argv=None):
    arguments = docopt.docopt(__doc__, options_first=True, argv=argv)
    corpusFileName = arguments['<corpusFile>']
    maxNumDocs = None
    if '--numDocs' in arguments: maxNumDocs = int(arguments['--numDocs'])
    updateHyperParams = arguments.get('--updateHyperParams', False)
    alpha = float(arguments.get('--alpha'))
    sigma = float(arguments.get('--sigma'))
    tau = float(arguments.get('--tau'))
    alphaF = float(arguments.get('--alphaF'))
    alphaTheta = float(arguments.get('--alphaTheta'))
    numIterationsPerRun = int(arguments.get('--numIterationsPerRun'))
    numTuningIterations = int(arguments.get('--numTuningIterations'))
    maxVocabSize = int(arguments.get('--vocabSize'))
    alphaFRangeStr = arguments.get('--alphaF_range')
    alphaThetaRangeStr = arguments.get('--alphaTheta_range')
    logF = arguments.get('--log', None)
    
    ###########################
    ## MAIN PROGRAM ###########
    ###########################
    
    startTime = time.time()
    
    
    customPrint(logF, "LOADING CORPUS", corpusFileName)
    customPrint(logF, "loading limited # docs:", maxNumDocs)
    lowercase = True
    customPrint(logF, "lowercasing:", lowercase)
    minTokenLen=0
    customPrint(logF, "removing tokens smaller than:", minTokenLen)
    removeStopWords=True
    customPrint(logF, "removing stopwords:", removeStopWords)
    customPrint(logF, "limit vocab size:", maxVocabSize)
    minNumTokens=10
    customPrint(logF, "remove tokens with frequency <", minNumTokens)
    customPrint(logF, "loading..")
    textCorpus = document_data.DocumentCorpus.loadFromCorpusFile(\
                            corpusFile=corpusFileName, 
                            maxNumDocs=maxNumDocs, 
                            lowercase=lowercase, 
                            minTokenLen=minTokenLen, 
                            removeStopWords=removeStopWords, 
                            maxVocabSize=maxVocabSize, 
                            minNumTokens=minNumTokens)
    trainCorpus, testCorpus = textCorpus.split()
    customPrint(logF, "done:")
    customPrint(logF, "loaded", len(textCorpus), "documents", "with a total of", textCorpus.getTotalNumWords(), "words (using half for testing)")
    customPrint(logF, "vocab size:", textCorpus.getVocabSize())
    
    numInitialTopics=10
    customPrint(logF, "# initial topics:", numInitialTopics)
    baseHyperParams = HyperParameters(alpha=alpha, sigma=sigma, tau=tau, alphaTheta=alphaTheta, 
                                          alphaF=alphaF, aGamma=1.0, bGamma=1.0)
    customPrint(logF, "(initial) hyperparameters:", baseHyperParams)
    
    customPrint(logF, "# sampling iterations per parameter combination:", numIterationsPerRun)
    customPrint(logF, "# iterations in Powell's search:", numTuningIterations)
     
    
    # determine choices for alphaF and alphaTheta
    varChoices = {}
    varChoices["alphaF"] = parseChoices(alphaFRangeStr)
    varChoices["alphaTheta"] = parseChoices(alphaThetaRangeStr)
    
    for tuningIter in range(numTuningIterations):
        for optimizedVar in ["alphaF", "alphaTheta"]:
            paramRetPairs = []
            for varVal in varChoices[optimizedVar]:
                seedRandom()
                curHyperParams = copy.deepcopy(baseHyperParams)
                setattr(curHyperParams, optimizedVar, varVal)
                ret = infer_topics.inferTopicsCollapsedGibbs(textCorpus=trainCorpus, 
                                           hyperParameters=curHyperParams, 
                                           numIterations=numIterationsPerRun, 
                                           numInitialTopics=numInitialTopics,
                                           updateHyperparameters=updateHyperParams,
                                           verbose=False,
                                           computeLogLikelihood=True,
                                           logLikelihoodEachIteration=False,
                                           estimatePerplexityForSplitCorpus=testCorpus,
                                           pplEachIteration=False)
                del ret['samplingVariables']
#                print (curHyperParams.__str__(), ret)
                paramRetPairs.append((curHyperParams, ret))
            bestParam = None
            bestPpl = float("inf")
            bestRet = None
            for (param, ret) in paramRetPairs:
                ppl = ret["perplexity"]
                if ppl < bestPpl:
                    bestParam = param
                    bestPpl = ppl
                    bestRet = ret
            customPrint(logF, "currently best params:", bestParam.__str__())
            baseHyperParams = bestParam
        customPrint(logF, "best ppl after tuning iteration", (tuningIter+1), ":", bestPpl)
    
    customPrint(logF, "best parameter combination found:", bestParam)
    customPrint(logF, "  with a PPL of", bestPpl)
    customPrint(logF, "  log likelihood was", bestRet["logLikelihood"])
    
    elapsedTime = (time.time() - startTime)
    customPrint(logF, "elapsed time:", elapsedTime, "seconds")

if __name__ == "__main__":
    sys.exit(main())
