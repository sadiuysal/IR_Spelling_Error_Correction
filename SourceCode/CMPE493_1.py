import collections
import re,sys
from itertools import combinations

Token = collections.namedtuple('Token', ['type', 'value', 'line', 'column'])
print("Running configuration:")
print(sys.argv)
corpusFile=sys.argv[1]#"corpus.txt"
errorsFile=sys.argv[2]#"spell-errors.txt"
testFileMisspelled=sys.argv[3]#"test-words-misspelled.txt"
fileForResults=sys.argv[4]#"results.txt"
smoothing=sys.argv[5]#True
#correctResults="test-words-correct.txt"  #for testing purposes






def tokenize(text): #tokenize given text with regex
    token_specification = [
        ('WORD',       r'[A-Za-z]([-\']){0,1}[A-Za-z]*'),    # regex for finding words
        ('NEWLINE',  r'\n'),           # regex for Line endings
        ('ABBREVIATION',       r'[A-Za-z]+([.]){1}(?=[ ]{1}[a-z])'), #I added this but found no abbreviation in dataset
    ]
    tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in token_specification)
    line_num = 1
    line_start = 0
    for mo in re.finditer(tok_regex, text):
        type = mo.lastgroup
        value = mo.group()
        column = mo.start() - line_start
        if type=='NEWLINE':
            line_start = mo.end()
            line_num += 1
            continue
        else:
            yield Token(type, value, line_num, column)
def createDict(corpusFile):  #creates dictionaries and fulfills
    f=open(corpusFile, "r")
    dict_tokens={} #stores token--frequency pairs
    dict_tokens_bigram={}  #stores bigram inverted index
    if f.mode == 'r':
        contents=f.read()
        for token in tokenize(contents):
            word=token.value.casefold()
            if not (dict_tokens.get(word) is None):
                dict_tokens[word]+=1
            else:
                dict_tokens[word]=1
                word="$"+word+"$"  #word for creating bigram dict.
                for i in range(len(word)-1):
                    bigram=word[i:i+2]
                    if not (dict_tokens_bigram.get(bigram) is None):
                        dict_tokens_bigram[bigram][0]+=1
                        dict_tokens_bigram[bigram][1].append(word[1:-1])
                    else:
                        dict_tokens_bigram[bigram]=[1,[]]
                        dict_tokens_bigram[bigram][1].append(word[1:-1])
    f.close()
    for token in dict_tokens:
        freq=dict_tokens[token]  #frequency of this token
        for i in range(len(token)-1):
            if i==0:   # since tokens starts with char and some errors contains deletion of first char,we count occurences of double char pair with " " such as " a"," b"
                doubleChar=" "+token[0]
                ind=indexHelper2(doubleChar)
                doubleCharCounts[ind]+=freq
            doubleChar=token[i:i+2]
            ind=indexHelper2(doubleChar)
            doubleCharCounts[ind]+=freq
        for c in token:
            ind=indexHelper(c)
            singleCharCounts[ind]+=freq
    return dict_tokens,dict_tokens_bigram


def generateBigrams(word):  #generates bigrams for given word
    bigrams=set()
    word="$"+word+"$"
    for i in range(len(word)-1):
        bigram=word[i:i+2]
        bigrams.add(bigram)
    return bigrams

def bigramIndexCandidates(bigram_set):  #finds candidates according to bigramIndex
    candidateSets={}  #dict of bigram's wordSet
    candidates=set() #set of candidate words according to minimum bigram check
    i=0
    for bigram in bigram_set:  #access each bigram's wordSet and store it in candidateSets
        i+=1
        if not (dict_tokens_bigram.get(bigram) is None):
            candidateSets[i]=dict_tokens_bigram[bigram][1]
        else:
            candidateSets[i]=[]
    #to find words that have at least len(candidateSets)*JC) number of same bigram
    for combination in list(combinations(candidateSets,int(len(candidateSets)*JC))):
        combinationSet=set(candidateSets[combination[0]])
        for i in range(len(combination)-1):
            combinationSet=combinationSet.intersection(set(candidateSets[combination[i+1]])) #intersect combination's sets
        candidates=candidates.union(combinationSet) # then union them
    return list(candidates)

def d_levenshtein(s1, s2):   #calculates d-levenshtein edit distance written via wikipedia pseudocode
    len_s1=len(s1)
    len_s2=len(s2)
    s1=" "+s1   #since in wikipedia strings are 1 indexed
    s2=" "+s2
    maxdist = len_s1+len_s2
    # d: dimensions length(s1)+1, length(s2)+1
    d=[[0 for i in range(len_s2+1)] for j in range(len_s1+1)]
    da={}
    d[-1][-1] = maxdist
    for i in range(len_s1+1):
        d[i][-1] = maxdist
        d[i][0] = i
    for j in range(1,len_s2+1):
        d[-1][j] = maxdist
        d[0][j] = j
    for i in range(1,len_s1+1):
            db = 0
            for j in range(1,len_s2+1):
                if not (da.get(s2[j]) is None):
                    k = da[s2[j]]
                else:
                    k = 0
                l = db
                if s1[i] == s2[j]:
                    cost = 0
                    db = j
                else:
                    cost = 1
                findmin=[d[i-1][j-1] + cost,  #substitution
                                   d[i][j-1] + 1,     #insertion
                                   d[i-1][j] + 1,     #deletion
                                   d[k-1][l-1] + (i-k-1) + 1 + (j-l-1)] #transposition
                d[i][j] = min(findmin)
            da[s1[i]] = i
    # Return last element and distance matrix
    return d[len_s1][len_s2] , d

def editDistCandidates(candidates,query):  #detect candidates with one edit dist. to query and returns
    newCandidates=set() #set of candidate words according to d_levenshtein edit distance check
    for candidate in candidates:
        #print(candidate)
        if d_levenshtein(candidate,query)[0] == 1:
            newCandidates.add(candidate)
    return newCandidates

def nofWordsCalculator():
    nofWords=0
    for value in dict_tokens.values():
        nofWords+=value
    return nofWords
def probPrior(word):  #returns probability of prior
    return dict_tokens.get(word)/nofWords

def probCalculator(editCount,count,word,smoothing,wWord):  #calculates noisy channel probability
    if smoothing:
        return ((editCount+1)/(count+alphabet_size+nof_symbols))*probPrior(word)
    else:
        if count==0:#unseen errors
            count=alphabet_size+nof_symbols
        return (editCount/count)*probPrior(word)


def analyzeDistMatrix(d,s1,s2):  #Analyze distance matrix and get the edit operations
    edits=[]
    i,j=len(s1),len(s2)
    s1=" "+s1
    s2=" "+s2
    while not(i==0 or j==0):
        curr=d[i][j]
        uCorner=d[i-1][j-1]
        left=d[i][j-1]
        upper=d[i-1][j]
        if uCorner<curr:
            edits.append(("sub",s1[i],s2[j]))  #s2[j] typed as s1[i]
            i+=-1
            j+=-1
        elif left<curr:
            edits.append(("ins",s1[i],s1[i]+s2[j]))  #s1[i] typed as s2[j-1:j+1]
            j+=-1
        elif upper<curr:
            edits.append(("del",s1[i-1:i+1],s1[i-1]))  #s1[i-1:i+1] typed as s2[j]
            i+=-1
        elif curr==left and curr==upper and curr==uCorner and curr>d[i-2][j-2]:
            edits.append(("trans",s1[i-1:i+1],s2[j-1:j+1]))  #s1[i-1:i+1] typed as s2[j-1:j+1]
            i+=-2
            j+=-2
        else:
            i+=-1
            j+=-1
    if not (i==0 and j==0):
        if j==0:
            while i!=0:
                edits.append(("del",s1[i-1:i+1],s1[i-1]))  #s1[i-1:i+1] typed as s2[j]
                i+=-1
        if i==0:
            while j!=0:
                edits.append(("ins",s1[i],s1[i]+s2[j]))  #s1[i] typed as s2[j-1:j+1]
                j+=-1
    return edits

def indexHelper(c):  #index finder for given char
    if c=="'":
        return alphabet_size   #I decided to use 26. index for ' sign
    elif c=="-":
        return alphabet_size+1   #27.index for - sign
    elif c==" ":
        return alphabet_size+2
    elif c==".":
        return alphabet_size+3
    else:
        return ord(c)-ord('a')
def indexHelper2(chars):   #index finder for given two chars
    ch1,ch2=chars[0],chars[1]
    return indexHelper(ch1)*(alphabet_size+nof_symbols-1)+indexHelper(ch2)

def fillConfMatrices(spellErrorsFile):  #fulfills the confusion matrices according to spell-errors.txt
    f=open(spellErrorsFile, "r")
    while True:
        line=f.readline()
        if not line:
            break
        c_word,words=parser(line)
        #print(c_word)
        #print(words)
        if c_word=="?":  #In spell-errors.txt ? means unknown correction
            continue
        for (word,count) in words:
            edits=analyzeDistMatrix(d_levenshtein(word,c_word)[1],word,c_word)
            for edit in edits:
                opr=edit[0]
                #print(edit)
                if opr=="sub":
                    rowInd=indexHelper(edit[1])
                    colInd=indexHelper(edit[2])
                    conf_sub[rowInd][colInd]+=count
                elif opr=="ins":
                    rowInd=indexHelper(edit[1])
                    colInd=indexHelper2(edit[2])
                    conf_ins[rowInd][colInd]+=count
                elif opr=="del":
                    rowInd=indexHelper2(edit[1])
                    colInd=indexHelper(edit[2])
                    conf_del[rowInd][colInd]+=count
                elif opr=="trans":
                    rowInd=indexHelper2(edit[1])
                    colInd=indexHelper2(edit[2])
                    conf_trans[rowInd][colInd]+=count
    f.close()

def parser(line):  #line parser for spell-errors.txt file
    line=line.rstrip()
    parts=line.split(': ')
    c_word=parts[0].lower()
    wordCandidates=parts[1].split(', ')
    words=[]
    for wordCandidate in wordCandidates:
        wordCandidate=wordCandidate.lower()
        ind=wordCandidate.find("*")
        if ind==-1:
            words.append((wordCandidate,1))
        else:
            words.append((wordCandidate[:ind],int(wordCandidate[ind+1:])))
    return c_word,words

def processQueries(queriesFile,fileToWrite,smoothing): #process all misspelled queries line by line and write corrected versions to fileToWrite
    f=open(queriesFile,"r")
    f2=open(fileToWrite,"w+")
    while True:
        query=f.readline().rstrip()
        if not query:
            break
        #if not (dict_tokens.get(query) is None):
            #if dict_tokens[query]>=2:
                #f2.write(query+"\n")
                #continue
        bigram_set=generateBigrams(query)
        firstCandidates=bigramIndexCandidates(bigram_set)
        candidates=editDistCandidates(firstCandidates,query)
        if len(candidates)==0:
            f2.write("\n")
            continue
        maxProb=0
        maxCandidate=""
        for candidate in candidates:
            d=d_levenshtein(query,candidate)[1]
            edit=analyzeDistMatrix(d,query,candidate)
            #print(query+">>"+candidate)
            #print(edit)
            edit=edit[0]
            opr=edit[0]
            if opr=="sub":
                rowInd=indexHelper(edit[1])
                colInd=indexHelper(edit[2])
                subCount=conf_sub[rowInd][colInd]
                count=singleCharCounts[rowInd]
                prob=probCalculator(subCount,count,candidate,smoothing,wWord=query)
            elif opr=="ins":
                rowInd=indexHelper(edit[1])
                colInd=indexHelper2(edit[2])
                insCount=conf_ins[rowInd][colInd]
                count=singleCharCounts[rowInd]
                prob=probCalculator(insCount,count,candidate,smoothing,wWord=query)
            elif opr=="del":
                rowInd=indexHelper2(edit[1])
                colInd=indexHelper(edit[2])
                delCount=conf_del[rowInd][colInd]
                count=doubleCharCounts[rowInd]
                prob=probCalculator(delCount,count,candidate,smoothing,wWord=query)
            elif opr=="trans":
                rowInd=indexHelper2(edit[1])
                colInd=indexHelper2(edit[2])
                transCount=conf_trans[rowInd][colInd]
                count=doubleCharCounts[rowInd]
                prob=probCalculator(transCount,count,candidate,smoothing,wWord=query)

            if prob>=maxProb:
                maxProb=prob
                maxCandidate=candidate

        f2.write(maxCandidate+"\n")
    f.close()
    f2.close()

def testTheProgram():
    f=open(fileForResults,'r')
    f2=open(correctResults,'r')
    f3=open("N-FAIL.txt","w+")
    f1=open(testFileMisspelled,"r")
    f4=open("FAIL.txt","w+")

    while True:
        l1=f.readline().strip()
        l2=f2.readline().strip()
        l3=f1.readline().strip()
        if (not l1) and (not l2) :
            break
        if l1 != l2:
            if dict_tokens.get(l2) is None:
                f3.write("DO NOT HAVE IN DICT--- "+" >>misspelledWord>> "+l3+" >>output>> "+l1+" >>trueForm>> "+l2+"\n")

            else:
                if d_levenshtein(l3,l2)[0]!=1:

                    f3.write("DO NOT HAVE 1 EDIT DIST--- "+" >>misspelledWord>> "+l3+" >>output>> "+l1+" >>trueForm>> "+l2+"\n")
                elif not (dict_tokens.get(l3) is None):

                    f3.write("HAVE MISSPELLED IN DICT--- "+" >>misspelledWord>> "+l3+" >>output>> "+l1+" >>trueForm>> "+l2+"\n")
                else:
                    f4.write("COULD NOT FIND TRUE CORRECTION--- "+" >>misspelledWord>> "+l3+" >>output>> "+l1+" >>trueForm>> "+l2+"\n")
    f.close()
    f2.close()
    f3.close()
    f1.close()
    f4.close()







JC=0.6 #for bigram set similarity
alphabet_size=26
nof_symbols=4 #  for my regex there is four extra symbol (- ,' , " ", . )
#confusion matrices
conf_del=[[0 for j in range(alphabet_size+nof_symbols)] for i in range((alphabet_size+nof_symbols)**2)]
conf_sub=[[0 for j in range(alphabet_size+nof_symbols)] for i in range(alphabet_size+nof_symbols)]
conf_ins=[[0 for j in range((alphabet_size+nof_symbols)**2)] for i in range(alphabet_size+nof_symbols)]
conf_trans=[[0 for j in range((alphabet_size+nof_symbols)**2)] for i in range((alphabet_size+nof_symbols)**2)]
#counts
singleCharCounts=[0 for i in range(alphabet_size+nof_symbols)]
doubleCharCounts=[0 for i in range((alphabet_size+nof_symbols)**2)]
#creating dictionaries and fill counts list
dict_tokens,dict_tokens_bigram=createDict(corpusFile)
nofWords=nofWordsCalculator()
#fill confusion matrices
fillConfMatrices(errorsFile)
#process queries
processQueries(queriesFile=testFileMisspelled,fileToWrite=fileForResults,smoothing=smoothing)
#testTheProgram()







