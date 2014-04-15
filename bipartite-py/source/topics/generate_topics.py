'''
Created on Mar 9, 2014

@author: Matthias Sperber
'''

import numpy as np
import source.prob as prob
from source.topics.infer_topics_hyperparam import HyperParameters
import source.expressions as expr
from source.document_data import *
from source.topics.state import GibbsSamplingVariables
import random

class BipartiteTopicGenerator(object):
    def generateTopics(self, vocabSize, numDocuments, numWordsPerDocument, hyperParameters):
        # need:
        # * vocabSize
        # * numDocuments
        # * numWordsPerDocument
        # * a_gamma, b_gamma => draw gammas
        
        
        dummyCorpus = DocumentCorpus(vocabSize=vocabSize)
        for _ in range(numDocuments):
            dummyCorpus.append(Document())
            for _ in range(numWordsPerDocument):
                dummyCorpus[-1].append(0)
        samplingVariables = GibbsSamplingVariables(textCorpus=dummyCorpus, vocabSize=vocabSize)
        
        
        samplingVariables.gammas = self.sampleGammas(vocabSize, hyperParameters.aGamma, hyperParameters.bGamma)
        
        K = []
        K.append(self.drawNumTopicsFirstWord(samplingVariables.gammas[0], hyperParameters))
        
        mj = [1] * int(K[-1])
        
        samplingVariables.zMat = np.zeros((vocabSize, K[0]))
        samplingVariables.uMat = np.zeros((vocabSize, K[0]))
        for iteratingTopic in range(int(K[0])):
            samplingVariables.zMat[0,iteratingTopic] = 1
            samplingVariables.uMat[0,iteratingTopic] = self.sampleNextU(iteratingWordType=0, 
                                                   iteratingTopic=iteratingTopic, 
                                                   u=samplingVariables.uMat, 
                                                   gammas=samplingVariables.gammas, 
                                                   mj=mj, 
                                                   hyperParameters=hyperParameters)
        
        for iteratingWordType in range(1, vocabSize):
            for iteratingTopic in range(int(K[iteratingWordType-1])):
                zij = self.sampleZEntry(curWordType=iteratingWordType, 
                                        curTopic=iteratingTopic,
                                        gammas=samplingVariables.gammas, 
                                        u=samplingVariables.uMat, 
                                        mj=mj, 
                                        hyperParameters=hyperParameters)
                samplingVariables.zMat[iteratingWordType, iteratingTopic] = zij
                if zij==1:
                    mj[iteratingTopic] += 1
            Kiplus = self.drawNumNewTopics(gammas=samplingVariables.gammas, 
                                           curWordType=iteratingWordType, 
                                           hyperParameters=hyperParameters)
            K.append(K[iteratingWordType-1] + Kiplus)
            samplingVariables.zMat = np.column_stack( [ samplingVariables.zMat , np.zeros((vocabSize,Kiplus))])
            samplingVariables.uMat = np.column_stack( [ samplingVariables.uMat , np.zeros((vocabSize,Kiplus))])
            mj += [1] * Kiplus
            for iteratingTopic in range(K[iteratingWordType-1], K[iteratingWordType]):
                samplingVariables.zMat[iteratingWordType,iteratingTopic] = 1
                samplingVariables.uMat[iteratingWordType,iteratingTopic] = self.sampleNextU(iteratingWordType=iteratingWordType, 
                                                   iteratingTopic=iteratingTopic, 
                                                   u=samplingVariables.uMat, 
                                                   gammas=samplingVariables.gammas, 
                                                   mj=mj, 
                                                   hyperParameters=hyperParameters)
        samplingVariables.activeTopics = range(K[-1])
        f = np.zeros(samplingVariables.zMat.shape)
        for iteratingTopic in range(K[-1]):
            fij = self.drawTopicWordDistribution(alphaF=hyperParameters.alphaF, 
                                            c_f=self.identity, 
                                            numTopics = mj[iteratingTopic])
            fij_index = 0
            for iteratingWordType in range(vocabSize):
                if samplingVariables.zMat[iteratingWordType, iteratingTopic] == 1:
                    f[iteratingWordType, iteratingTopic]  = fij[fij_index]
                    fij_index += 1
    
        theta = []
        for _ in range(numDocuments):
            theta.append(self.drawDocumentTopics(alphaTheta=hyperParameters.alphaTheta, 
                                                 c_theta=self.identity, numTopics = K[-1]))
#            theta.append([1.0/K[-1]] * K[-1])
            if K[-1]>0 and not 0.99 < np.sum(theta[-1]) < 1.0001:
                print theta[-1]
                assert 0.99 < np.sum(theta[-1]) < 1.0001
#            
        docs = DocumentCorpus(vocabSize=vocabSize)
        emptyWord = -1
        if K[-1]==0:
            raise NoTopicsException()
        for iteratingDocNo in range(numDocuments):
            docs.append(Document())
            for iteratingWordPos in range(numWordsPerDocument):
                tlk = self.drawTopicForWord(topicProportions=theta[iteratingDocNo])
                samplingVariables.tLArr[iteratingDocNo][iteratingWordPos] = tlk
                if mj[tlk]==0:
                    docs[-1].append(emptyWord)
                else:
                    docs[-1].append(self.drawWordFromTopic(wordDistribution=f[:,tlk].transpose()))
                    
        samplingVariables.textCorpus = docs
        samplingVariables.initCounts(docs)
        return samplingVariables
    
    def identity(self, x): return x
    
    def sampleGammas(self, vocabSize, aGamma, bGamma):
        return [self.sampleOneGamma(aGamma, bGamma) for _ in range(vocabSize)]
    
    def sampleOneGamma(self, aGamma, bGamma):
        return np.random.gamma(aGamma, 1/bGamma)
    
    def drawNumTopicsFirstWord(self, gamma0, hyperParameters):
        return np.random.poisson(expr.psiFunction(gamma0, 
                                                  hyperParameters.alpha, 
                                                  hyperParameters.sigma, 
                                                  hyperParameters.tau), 
                                 (1))[0]
    def drawNumNewTopics(self, gammas, curWordType, hyperParameters):
        return np.random.poisson(expr.psiTildeFunction(gammas[curWordType], 
                                                       sum([gammas[k] for k in range(curWordType)]), 
                                                       hyperParameters.alpha, 
                                                       hyperParameters.sigma, 
                                                       hyperParameters.tau), 
                                 (1))[0]
        
    def sampleNextU(self, iteratingWordType, iteratingTopic, u, gammas, mj, hyperParameters):
        return prob.sampleFrom15(gammas[:iteratingWordType+1], u[:iteratingWordType,iteratingTopic], mj[iteratingTopic], hyperParameters)
    
    def sampleZEntry(self, curWordType, curTopic, gammas, u, mj, hyperParameters):
        gammaUSum = sum([gammas[iteratingWordType]*u[iteratingWordType,curTopic] for iteratingWordType in range(curWordType)])
        probability = 1.0 - (expr.kappaFunction(mj[curTopic], hyperParameters.tau + gammas[curWordType] + gammaUSum, hyperParameters.alpha, hyperParameters.sigma, hyperParameters.tau) \
                    / expr.kappaFunction(mj[curTopic], hyperParameters.tau + gammaUSum, hyperParameters.alpha, hyperParameters.sigma, hyperParameters.tau))
        if prob.flipCoin(probability):
            return 1
        else:
            return 0
    
    def drawTopicWordDistribution(self, alphaF, c_f, numTopics):
        return np.random.dirichlet([alphaF/c_f(numTopics) for _ in range(numTopics)], 1).transpose()
    
    def drawDocumentTopics(self, alphaTheta, c_theta, numTopics):
        return np.random.dirichlet([alphaTheta/c_theta(numTopics) for _ in range(numTopics)], 1)[0]
    
    def drawTopicForWord(self, topicProportions):
        try:
            return np.nonzero(np.random.multinomial(1, topicProportions, size=1))[1][0]
        except IndexError:
            print "breakpoint"
    def drawWordFromTopic(self, wordDistribution):
        return np.nonzero(np.random.multinomial(1, wordDistribution, size=1))[1][0]

class NoTopicsException(Exception):
    pass

#np.random.seed(15)
#random.seed(15)
#
#hyperParameters = HyperParameters(alpha=5.0, sigma=0.5, tau=1.0, alphaTheta=1.0, 
#                                          alphaF=1.0, aGamma=1.0, bGamma=1.0)
#BipartiteTopicGenerator().generateTopics(vocabSize=15, 
#                                         numDocuments=6, 
#                                         numWordsPerDocument=5, 
#                                         hyperParameters=hyperParameters)