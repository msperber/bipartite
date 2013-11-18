'''
Created on Nov 11, 2013

@author: Matthias Sperber
'''

from numpy.random import poisson
import math
from source.prob import sampleFrom15
import source.expressions as expr    

def selectBooksForFirstReader(gammas, simulationParams, **kw):
    '''
        selects a number of books, parametrized by lambda_1 ("interest in reading")
        returns dictionary: book_no => book_score (bookScores[j] = x_{1j})
    '''
    # for easy unit testing, these can be replaced by dummies:
    if 'poissonFunction' in kw: poisson = kw['poissonFunction']
    if 'sampleFrom15Function' in kw: sampleFrom15 = kw['sampleFrom15Function']
    
    numBooks = poisson(expr.psiFunction(gammas[0], simulationParams), (1)) # TODO: verify (not sure what psi SUBSCRIPT GAMMA means here..)
    bookScores = {}
    for j in range(numBooks):
        bookScores[j] = sampleFrom15([gammas[0]], [], 0, simulationParams) # TODO: check

    return numBooks, bookScores

def uScoreFromXScore(xScore):
    return math.exp(-xScore)

def selectBooksForIthReader(gammas, numBooks, prevBookScoreList, simulationParameters):
    tao = simulationParameters.tao
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
        prob = 1.0 - (expr.kappaFunction(numTimesRead[j], tao + gammas[curReaderI] + gammaUSum, simulationParameters) \
                    / expr.kappaFunction(numTimesRead[j], tao + gammaUSum, simulationParameters))
        if prob.flipCoin(prob):
            bookScores[j] = 0.0
    
    # consider unread books:
    numAdditionalBooks = poisson(expr.psiTildeFunction(gammas[curReaderI], sum([gammas[k] for k in range(curReaderI)]), simulationParameters), (1)) # TODO: sv: should be fine now double check
    for j in range(numBooks, numBooks+numAdditionalBooks):
        bookScores[j] = 0.0
    
    # assign scores
    for bookNo in bookScores:
        uList = [uScoreFromXScore(prevBookScoreList[k-1].get(j, 0.0)) for k in range(curReaderI)] # if book has not been read score is zero.
        bookScores[bookNo] = sampleFrom15(gammas[:curReaderI+1], uList, numTimesRead[bookNo], simulationParameters) # TODO: check
    
    return numBooks + numAdditionalBooks, bookScores

def generateBipartiteGraph(simulationParameters, gammas):
    numBooks, firstScores = selectBooksForFirstReader(gammas, simulationParameters)
    allScores = [firstScores]
    for _ in range(len(gammas)-1):
        numBooks, scores = selectBooksForIthReader(gammas, numBooks, allScores, simulationParameters)
    sparseMatrix = []
    for scores in allScores:
        sortedBookNumbers = sorted(scores.keys())
        sparseMatrix.append(sortedBookNumbers)
    return sparseMatrix
