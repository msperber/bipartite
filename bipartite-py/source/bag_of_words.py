'''
Created on Dec 1, 2013

@author: Matthias Sperber
'''

import string
import re
import operator

alnum = set(string.letters + string.digits)

# English stopwords from http://www.lextek.com/manuals/onix/stopwords2.html
stopwords = set(["a","a's","able","about","above","according","accordingly","across","actually","after","afterwards","again","against","ain't","all","allow","allows","almost","alone","along","already","also","although","always","am","among","amongst","an","and","another","any","anybody","anyhow","anyone","anything","anyway","anyways","anywhere","apart","appear","appreciate","appropriate","are","aren't","around","as","aside","ask","asking","associated","at","available","away","awfully","b","be","became","because","become","becomes","becoming","been","before","beforehand","behind","being","believe","below","beside","besides","best","better","between","beyond","both","brief","but","by","c","c'mon","c's","came","can","can't","cannot","cant","cause","causes","certain","certainly","changes","clearly","co","com","come","comes","concerning","consequently","consider","considering","contain","containing","contains","corresponding","could","couldn't","course","currently","d","definitely","described","despite","did","didn't","different","do","does","doesn't","doing","don't","done","down","downwards","during","e","each","edu","eg","eight","either","else","elsewhere","enough","entirely","especially","et","etc","even","ever","every","everybody","everyone","everything","everywhere","ex","exactly","example","except","f","far","few","fifth","first","five","followed","following","follows","for","former","formerly","forth","four","from","further","furthermore","g","get","gets","getting","given","gives","go","goes","going","gone","got","gotten","greetings","h","had","hadn't","happens","hardly","has","hasn't","have","haven't","having","he","he's","hello","help","hence","her","here","here's","hereafter","hereby","herein","hereupon","hers","herself","hi","him","himself","his","hither","hopefully","how","howbeit","however","i","i'd","i'll","i'm","i've","ie","if","ignored","immediate","in","inasmuch","inc","indeed","indicate","indicated","indicates","inner","insofar","instead","into","inward","is","isn't","it","it'd","it'll","it's","its","itself","j","just","k","keep","keeps","kept","know","knows","known","l","last","lately","later","latter","latterly","least","less","lest","let","let's","like","liked","likely","little","look","looking","looks","ltd","m","mainly","many","may","maybe","me","mean","meanwhile","merely","might","more","moreover","most","mostly","much","must","my","myself","n","name","namely","nd","near","nearly","necessary","need","needs","neither","never","nevertheless","new","next","nine","no","nobody","non","none","noone","nor","normally","not","nothing","novel","now","nowhere","o","obviously","of","off","often","oh","ok","okay","old","on","once","one","ones","only","onto","or","other","others","otherwise","ought","our","ours","ourselves","out","outside","over","overall","own","p","particular","particularly","per","perhaps","placed","please","plus","possible","presumably","probably","provides","q","que","quite","qv","r","rather","rd","re","really","reasonably","regarding","regardless","regards","relatively","respectively","right","s","said","same","saw","say","saying","says","second","secondly","see","seeing","seem","seemed","seeming","seems","seen","self","selves","sensible","sent","serious","seriously","seven","several","shall","she","should","shouldn't","since","six","so","some","somebody","somehow","someone","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specified","specify","specifying","still","sub","such","sup","sure","t","t's","take","taken","tell","tends","th","than","thank","thanks","thanx","that","that's","thats","the","their","theirs","them","themselves","then","thence","there","there's","thereafter","thereby","therefore","therein","theres","thereupon","these","they","they'd","they'll","they're","they've","think","third","this","thorough","thoroughly","those","though","three","through","throughout","thru","thus","to","together","too","took","toward","towards","tried","tries","truly","try","trying","twice","two","u","un","under","unfortunately","unless","unlikely","until","unto","up","upon","us","use","used","useful","uses","using","usually","uucp","v","value","various","very","via","viz","vs","w","want","wants","was","wasn't","way","we","we'd","we'll","we're","we've","welcome","well","went","were","weren't","what","what's","whatever","when","whence","whenever","where","where's","whereafter","whereas","whereby","wherein","whereupon","wherever","whether","which","while","whither","who","who's","whoever","whole","whom","whose","why","will","willing","wish","with","within","without","won't","wonder","would","would","wouldn't","x","y","yes","yet","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves","z","zero"])

class BagOfWords(dict):
    def __init__(self, wordCounts = {}):
        self.update(wordCounts)
    def numWords(self):
        return sum([self[i] for i in self.keys()])
    def getContainedWords(self):
        return self.keys()
        
class DocumentCorpus(list):
    """
    a corpus is a list of bags of words (access documents like with a normal python list)
    """
    def __init__(self, documents=[], vocab=None):
        self.extend(documents)
        self.vocab = vocab
    def getVocabList(self):
        if self.vocab is None:
            raise Exception("no vocab was specified")
        else:
            return self.vocab
    
    def computeSplitCorpus(self, ratio):
        """
        splits the corpus vertically into two: each new corpus will have the same
        list of documents as before, but words in each document will be split randomly
        between both topics, according to the given ratio   
        """
        # TODO: implement
        raise NotImplementedError()
    @staticmethod
    def loadFromDatFile(datFile, vocabFile=None):
        """
        assumes this format:
        3 1:2 5:1
        1 4:1
        
        optionally, a corresponding vocab file can be specified
        """
        vocab = None
        if vocabFile is not None:
            vocab = [line.strip() for line in open(vocabFile).readlines()]
        
        documents=[]
        for line in open(datFile):
            spl = line.split()
            assert len(spl)>=1
            numWords = int(spl[0])
            wordCounts = {}
            for wordCountStr in spl[1:]:
                wordIndex, count = tuple([int(c) for c in wordCountStr.split(":")])
                assert count>0
                wordCounts[wordIndex] = count
            bagOfWords = BagOfWords(wordCounts)
            assert numWords==bagOfWords.numWords()
            documents.append(bagOfWords)
        return DocumentCorpus(documents=documents, vocab=vocab)
    
    @staticmethod
    def isValidToken(token, minTokenLen=1, removeStopWords = False):
        return len(set(token) & alnum) > 0 \
            and len(token)>=minTokenLen \
            and (not removeStopWords or token.lower() not in stopwords)
    @staticmethod
    def normalizeToken(token, lowercase=False):
        retToken = re.sub(r"^\W+|\W+$", "", token)
        if lowercase:
            retToken = retToken.lower()
        return retToken
    @staticmethod
    def loadFromCorpusFile(corpusFile,
                           maxNumDocs = None,
                           lowercase=False,
                           minTokenLen = 1,
                           removeStopWords = False,
                           maxVocabSize=None,
                           minNumTokens = 1
                           ):
        
        # first, build the vocabular
        allWords = []
        numDocs = 0
        for line in open(corpusFile):
            if maxNumDocs is not None and numDocs>= maxNumDocs:
                break
            doc = [n \
                     for n in [DocumentCorpus.normalizeToken(t, lowercase) for t in line.split()] \
                     if DocumentCorpus.isValidToken(n, minTokenLen=minTokenLen, 
                                                    removeStopWords=removeStopWords)]
            allWords += doc
            numDocs += 1
        vocabCounts = {}
        for word in allWords:
            vocabCounts[word] = vocabCounts.get(word,0) + 1
        vocab = [x[0] for x in reversed(sorted(vocabCounts.iteritems(), key=operator.itemgetter(1)))]
        if maxVocabSize is not None and len(vocab)>maxVocabSize:
            vocab = vocab[:maxVocabSize]
        vocab = set(vocab)
        
        # second, build the actual documents
        tokenSequences = []
        for line in open(corpusFile):
            if maxNumDocs is not None and len(tokenSequences)>= maxNumDocs:
                break
            doc = [n \
                     for n in [DocumentCorpus.normalizeToken(t, lowercase) for t in line.split()] \
                     if DocumentCorpus.isValidToken(n, minTokenLen=minTokenLen, 
                                                    removeStopWords=removeStopWords) \
                            and n in vocab]
            if len(doc) >= minNumTokens:
                tokenSequences.append(doc)

        vocab = sorted(vocab)
        bagsOfWords = []
        for seq in tokenSequences:
            curBag = BagOfWords()
            for token in seq:
                tokenIndex = vocab.index(token)
                curBag[tokenIndex] = curBag.get(tokenIndex, 0) + 1
            bagsOfWords.append(curBag)
        return DocumentCorpus(documents=bagsOfWords, vocab=vocab)
    
    
    