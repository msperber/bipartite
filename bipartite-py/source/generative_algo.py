'''
Created on Nov 11, 2013

@author: Matthias Sperber
'''

#from numpy.random import poisson
import numpy
import math
import source.prob as prob
import source.expressions as expr    

def selectBooksForFirstReader(gammas, simulationParams,
                              # override for module tests: 
                              poisson=numpy.random.poisson, sampleFrom15=prob.sampleFrom15):
    '''
        selects a number of books, parametrized by lambda_1 ("interest in reading")
        returns new number of read books,
                dictionary: book_no => book_score (bookScores[j] = x_{1j})
    '''
    
    numBooks = poisson(expr.psiFunction(gammas[0], simulationParams), (1))
    bookScores = {}
    for j in range(numBooks):
        bookScores[j] = sampleFrom15([gammas[0]], [], 0, simulationParams)

    return numBooks, bookScores

def uScoreFromXScore(xScore):
    return math.exp(-xScore)

def selectBooksForIthReader(gammas, numBooks, prevBookScoreList, simulationParameters,
                            poisson=numpy.random.poisson, sampleFrom15=prob.sampleFrom15):
    tau = simulationParameters.tau
    curReaderI = len(prevBookScoreList) # (0-based reader index)
    
    numTimesRead = [] # m_j
    for j in range(numBooks):
        numTimesRead.append(0)
        for scores in prevBookScoreList:
            if j in scores:
                numTimesRead[-1] += 1
    
    bookScores = {}
    # consider previously read books:
    for j in range(numBooks):
        gammaUSum = sum([gammas[k]*uScoreFromXScore(prevBookScoreList[k-1].get(j, 0.0)) for k in range(curReaderI)])
        probability = 1.0 - (expr.kappaFunction(numTimesRead[j], tau + gammas[curReaderI] + gammaUSum, simulationParameters) \
                    / expr.kappaFunction(numTimesRead[j], tau + gammaUSum, simulationParameters))
        if prob.flipCoin(probability):
            bookScores[j] = 0.0
    
    # consider unread books:
    numAdditionalBooks = poisson(expr.psiTildeFunction(gammas[curReaderI], sum([gammas[k] for k in range(curReaderI)]), simulationParameters), (1)) # TODO: sv: should be fine now double check
    for j in range(numBooks, numBooks+numAdditionalBooks):
        bookScores[j] = 0.0
    
    # assign scores
    for bookNo in bookScores:
        uList = [uScoreFromXScore(prevBookScoreList[k-1].get(j, 0.0)) for k in range(curReaderI)] # if book has not been read score is zero.
        bookScores[bookNo] = sampleFrom15(gammas[:curReaderI+1], uList, numTimesRead[bookNo] if bookNo < len(numTimesRead) else 0, simulationParameters)
    
    return numBooks + numAdditionalBooks, bookScores

def generateBipartiteGraph(simulationParameters, gammas):
    numBooks, firstScores = selectBooksForFirstReader(gammas, simulationParameters)
    allScores = [firstScores]
    for _ in range(len(gammas)-1):
        numBooks, scores = selectBooksForIthReader(gammas, numBooks, allScores, simulationParameters)
        allScores.append(scores)
    sparseMatrix = []
    for scores in allScores:
        sortedBookNumbers = sorted(scores.keys())
        sparseMatrix.append(sortedBookNumbers)
    return allScores, sparseMatrix
