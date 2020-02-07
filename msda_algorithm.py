def readNTable(fileName):
    nTable = {}
    f = open(fileName, "r")
    for line in f:
        #line = f.readline()
        line = line.split(" ")
        if line[0] not in nTable:
            nTable[line[0]] = []
        nTable[line[0]].append(line[1].strip())
    return nTable

def readNTableForObama(fileName):
    nTable = {}
    f = open(fileName, "r")
    for line in f:
        #line = f.readline()
        line = line.split(" ")
        if line[0] not in nTable:
            nTable[line[0]] = []
        if line[1].strip() not in nTable:
            nTable[line[1].strip()] = []
        nTable[line[0]].append(line[1].strip())
        nTable[line[1].strip()].append(line[0])
    return nTable

def msda(nTable, nodes):
    count = {}
    score = {}
    c=1
    for key in nTable:
        count[key] = len(nTable[key])
        score[key] = 1
    #print(count)
    #return
    while len(count) > nodes:
        for x in range(c):
            #print(""+str(c)+" "+str(x))
            ex = []
            for node in count:
                if count[node] == c and score[node] == x:
                    ex.append(node)
            #print("EX:")
            #print(ex)
            executeRemoves(nTable, ex, count, score)
            #print(nTable)
            #print(count)
            #print(score)
            #print("!!!!!!!!!!   "+str(c))
            if len(count) <= nodes:
                print(count)
                print(score)
                return
        for x in range(c):
            ex = []
            for node in count:
                if count[node] == x and score[node] == c:
                    ex.append(node)
            #print("EX:")
            #print(ex)
            executeRemoves(nTable, ex, count, score)
            #print(nTable)
            #print(count)
            #print(score)
            #print("@@@@@@@   "+str(c))
            if len(count) <= nodes:
                print(count)
                print(score)
                return
        ex = []
        for node in count:
            if count[node] == c and score[node] == c:
                ex.append(node)
        #print("EX:")
        #print(ex)
        executeRemoves(nTable, ex, count, score)
        #print(nTable)
        #print(count)
        #print(score)
        #print("WWWW   "+str(c))
        if len(count) <= nodes:
            print(count)
            print(score)
            return
        c = c+1
        

    print(count)
    print(score)

def executeRemoves(nTable, ex, count, score):
    for node in ex:
        for add in nTable[node]:
            if score[node] == 0:
                score[node] = 1
            if add in score:
                score[add] = score[add] + score[node]
                count[add] = count[add] -1
        nTable.pop(node)
        count.pop(node)
        score.pop(node)
        
        
# ======================================================================================================================
# START
# ======================================================================================================================
        
print("Hello world")
nTable = readNTableForObama("obama.txt")
msda(nTable, 7)
#print(nTable["8"])
#print(nTable["7"])