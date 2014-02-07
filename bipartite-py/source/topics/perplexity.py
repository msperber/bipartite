'''
Created on Feb 7, 2014

@author: Matthias Sperber
'''

import math

import source.utility as utility

def computePerplexityEstimateForWord(docNo, wordType, samplingVariables, hyperParameters):
    val = 0.0
    for iteratingTopic in samplingVariables.getActiveTopics():
        # product will 0 if z[]=0, so we can skip computing the rest:
        if utility.approx_equal(samplingVariables.zMat[wordType][iteratingTopic], 0.0):
            continue
        else: 
            factor1 = hyperParameters.alphaTheta/len(samplingVariables.getActiveTopics()) \
                + samplingVariables.counts.numTopicOccurencesInDoc.get((docNo, iteratingTopic),0)
            factor2 = 1.0 # already checked this
            factor3 = hyperParameters.alphaF\
                     / samplingVariables.counts.numWordTypesActivatedInTopic.get(iteratingTopic,0)\
                     + samplingVariables.counts.numTopicAssignmentsToWordType.get(wordType,0)
            val += factor1 * factor2 * factor3
    return val
            

def updatePerplexityWordAvg(perplexityWordAvg, iteration, samplingVariables, hyperParameters, 
                            splitCorpus):
    for docNo in range(len(splitCorpus)):
        for wordPos in range(len(splitCorpus[docNo])):
            wordType = splitCorpus[docNo][wordPos]
            perplexityWordAvg[(docNo,wordPos)] = \
                    1.0 * iteration / (iteration+1) * perplexityWordAvg.get((docNo, wordType), 0) \
                    + 1.0 / (iteration+1) * computePerplexityEstimateForWord(
                                                            docNo=docNo,
                                                            wordType=wordType,
                                                            samplingVariables=samplingVariables,
                                                            hyperParameters=hyperParameters)

def computeTotalPerplexityFromWordAvg(perplexityWordAvg):
    sum = 0.0
    for docWordPos in perplexityWordAvg:
        sum += math.log(perplexityWordAvg[docWordPos])
    return math.exp( - sum / len(perplexityWordAvg))
    
    