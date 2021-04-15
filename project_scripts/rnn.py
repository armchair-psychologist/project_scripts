"""
Need to set up the pipeline which is
1.) get dataframes ready
2.) pass data to baseline model. Save model after training
3.) pass data with specific words taken out to model. Save model after training
4.) pass data to model for permutation testing. Save graph after training
"""

"""
need to open 3 threads for training/testing 3 different models
a single accuracy will be a list of 5 accuracies one for each model
"""
from copy import deepcopy
from threading import Thread
import pickle
from queue import Queue
from utilities import *
from pandas import DataFrame
import tensorflow as tf
from statsmodels.stats.weightstats import ttest_ind
import numpy as np
import matplotlib as plt
def createNormalDataSet():
   '''
   this must produce a balancer
   '''
   file1 = CsvToDf("../data/typed_posts.csv",batchSize=400)
   com = Combiner([file1],[["type","title"]])
   balan = Balancer(com,300,100)
   balan.createDataSets()
   train = balan.getTrainSet()
   test = balan.getTestSet()
   return train,test

def removeWord(text,toRemove):
    words = text.split()
    cleanWords = [i for i in words if i not in toRemove]
    return " ".join(cleanWords)
def createVariantDataset(df,toRemove):
    # the dataframe must be the output of createNormalDataSet
    #this must produce dataframe
    return df.apply(removeWord,args=(toRemove))
    
def permuationTrain(trainData, testData, q):
    # this needs to train on base 100 times and not save the models
    #must return a list of 5 accuracies
    out = [[],[],[],[],[]]
    count = 0
    tTrain = tokenize(trainData["posts"])
    dTest = tokenize(testData["posts"])
    dLabel = get4Dim(testData["type"])
    labelCopy = deepcopy(get4Dim(trainData["type"]))
    while count < 100:
        for idx,i in enumerate(labelCopy): #this will shuffle the labels
            np.random.shuffle(labelCopy[idx])
        models = train4Dim(tTrain,labelCopy,5)
        acc = getAccuracy(models,dTest,dLabel) + getTotalAccuracy(models,dTest,dLabel)
        for idx,i in enumerate(acc):
            out[idx].append(i)
    q.put({"perm":out})
def createDimModel():
    vocab_size = 10000
    max_length = 2016
    embedding_dim = 25
    return tf.keras.Sequential([ 
                            tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
                            tf.keras.layers.GRU(64, return_sequences=True),
                            tf.keras.layers.SimpleRNN(64),
                            tf.keras.layers.Dense(1, activation='sigmoid'),])

def tokenize(postsSet):
    tokenizer = Tokenizer(num_words = vocab_size, oov_token = "<OOV>")
    tokenizer.fit_on_texts(postsSet)
    training_sequences = tokenizer.texts_to_sequences(postsSet)
    training_padded = pad_sequences(training_sequences, padding = 'post', maxlen = 2016)
    # training_sequences = np.array(training_sequences)
    training_padded = np.array(training_padded)
    return training_padded

def get4Dim(df):
    personality_dict = {"ENTJ" : 0, "INTJ" : 0, "ENTP" : 0, "INTP" : 0, "INFJ" : 0, "INFP" : 0, "ENFJ" : 0, 
                    "ENFP" : 0, "ESTP" : 0, "ESTJ" : 0, "ISTP" : 0, "ISTJ" : 0, "ISFJ" : 0, "ISFP" : 0, 
                    "ESFJ" : 0, "ESFP" : 0}
    out = [[0 for i in range(len(df.index))],[0 for i in range(len(df.index))],[0 for i in range(len(df.index))],[0 for i in range(len(df.index))]]
    for idx,row in enumerate(df):
        personality = row
        if isinstance(personality,str) and personality in personality_dict:
            personality = personality.upper()
            if personality[0] == "E":
                out[0][idx] = 1
            if personality[1] == "S":
                out[1][idx] = 1
            if personality[2] == "T":
                out[2][idx] = 1
            if personality[3] == "J":
                out[3][idx] = 1
    return out

def trainBase(trainData, testData, q):
    # this just needs to do a basic train and save model
    # must return a list of 5 accuracies
    models = train4Dim(tokenize(trainData["posts"]),get4Dim(trainData["type"]),5)
    acc = getAccuracy(models,tokenize(testData["posts"]),get4Dim(testData["type"])) + getTotalAccuracy(models,tokenize(testData["posts"]),get4Dim(testData["type"]))
    q.put({"base":acc})
def train4Dim(trainPost,trainLabels,num_epochs):
    models = [createDimModel(),createDimModel(),createDimModel(),createDimModel()]
    for idx,dims in enumerate(trainLabels):
        print(f"training dim {idx+1}")
        models[idx].compile(loss = tf.keras.losses.BinaryCrossentropy(), optimizer = 'adam', metrics = ["accuracy"])
        models[idx].fit(trainPost, np.array(trainLabels[idx]), epochs = num_epochs, verbose = 1)
    return models
def getTotalAccuracy(models,testing_padded,testing_labels):
    total = None
    isEmpty = True
    for idx,model in enumerate(models):
        test = testing_labels[idx]
        modelOut = np.squeeze(np.round(models[idx].predict(testing_padded)))
        if isEmpty:
            total = np.array(modelOut)
            isEmpty = False
        else:
            total = np.column_stack((total,modelOut))
    labels = None
    isEmpty = True
    for idx,col in enumerate(testing_labels):
        if isEmpty:
            labels = np.array(col)
            isEmpty = False
        else:
            labels = np.column_stack((labels,col))
    return np.mean(np.sum(abs(total-labels),axis=1) == 0)

def getAccuracy(models,testing_padded,testing_labels):
    out = []
    for idx,model in enumerate(models):
        test = testing_labels[idx]
        modelOut = np.round(models[idx].predict(testing_padded))
        out.append(np.mean(abs(np.squeeze(modelOut)-np.squeeze(test)) == 0))
    return out
def trainVariant(trainData, testData, q,num):
    # must return a list of 5 accuracies
    models = train4Dim(tokenize(trainData["posts"]),get4Dim(trainData["type"]),5)
    acc = getAccuracy(models,tokenize(testData["posts"]),get4Dim(testData["type"])) + getTotalAccuracy(models,tokenize(testData["posts"]),get4Dim(testData["type"]))
    q.put({f"variant {num}":acc})


def saveModel(name,model):
    #must save a model
    model.save(f"model/{name}")

def writeToFile(aString, fName):
    #write to a file
    with open(fName,"a") as file:
        file.write(aString)


def compAccuracy(aList):
    #identify which accuracy the model belongs to
    #return a tuple (basAcc,varAcc)
    out = [0 for i in range(aList)]
    for i in aList:
        if "variant 1" in i:
            out[1] = i["variant 1"] 
        elif "variant 2" in i:
            out[2] = i["variant 2"]
        else:
            out[0] = i["base"]
    return tuple(out)
def ttest(permDist, acc):
    out = []
    for idx,i in enumerate(permDist):
        out.append(ttest_ind(np.array(i),np.array([acc[idx]]))[1])
    return out
def graph(x,y,xLabel,yLabel,title,figname):
    plt.clf()
    plt.hist(x,color="c",edgecolor="k",alpha=0.5)
    plt.axvline(np.array(x).mean(),color="k",linestyle="dashed",linewidth=3,label="average")
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title)
    
    yAxis = np.arange(0,10,1)
    acRes = [y]
    z = np.array(acRes*10)
    plt.plot(z,yAxis,label="model accuracy")
    p_value = ttest_ind(x,[y])[1]
    plt.plot([],[],label=f"p-value: {np.round(p_value,4)}",color="w")
    plt.legend()
    plt.savefig(figname)

def graphPermRes(permDist, accuracyList, names):
    for idx,i in accuracyList:
        graph(permDist[0],i[0],"accuracy","count","Introvert/Extrovert prediction",f"{names[idx]} dim1.png")
        graph(permDist[1],i[1],"accuracy","count","Sensors/Intuitives prediction",f"{names[idx]} dim2.png")
        graph(permDist[2],i[2],"accuracy","count","Thinkers/Feelers prediction",f"{names[idx]} dim3.png")
        graph(permDist[3],i[3],"accuracy","count","Judgers/Perceivers prediction",f"{names[idx]} dim4.png")
        graph(permDist[4],i[4],"accuracy","count","Personality classification",f"{names[idx]} total.png")

if __name__ == "__main__":
    q = Queue()
    trainSet,testSet = createNormalDataSet()#second parameter is the limit of training data that can be used
    varSet = createVariantDataset(trainSet,['loyalty:', 'dogmas','post.]', "'rarity',",'static:', 'grins.','358890', 'intimidate,','84389', '84390','beck.', 'alpha;','transversal', 'ri,'
          ,'gosh....this', 'now.....but','error(when', 'times(asshole','oooooh...', 'floss?','rupp,', 'customization.',
          'backstories', 'error!!!,','time...there', 'yeah...this','jimmers,', 'boat?...','fi>ti>fe>te', '-plans','hulme', 'scotland?'])
    varSet2 = createVariantDataset(trainSet,['sweet', 'deal', 'month', 'then', 'know', 'thought', 'up.', 'off', 'asking', 'he', 'idea.', 'crush', 'this', 'college', 'it', 'most', 'dealing', 'late', 'key', 'his', 'talk', 'oh', 'those', 'everything.'])
    base = Thread(target=trainBase, args=(trainSet, testSet, q))
    var = Thread(target=trainBase, args=(varSet, testSet, q))
    var2 = Thread(target=trainBase, args=(varSet2, testSet, q))
    perm = Thread(target=permuationTrain, args=(trainSet, testSet, q))
    base.start()
    var.start()
    var2.start()
    perm.start()
    base.join()
    var.join()
    var2.join()
    res1 = q.get()
    res2 = q.get()
    res3 = q.get()
    baseAcc, varAcc, varAcc2 = compAccuracy([res1, res2,res3])
    writeToFile(f"accuracy for base: {baseAcc}", "out.txt")
    writeToFile(f"accuracy for unique words: {varAcc}", "out.txt")
    writeToFile(f"accuracy for common words: {varAcc2}", "out.txt")
    perm.join()
    dist = q.get()
    with open("permDist", "wb") as file:
        pickle.dump(file, dist)
    writeToFile(f"p-value for base: {ttest(dist,baseAcc)}", "out.txt")
    writeToFile(f"p-value for unique words: {ttest(dist,varAcc)}", "out.txt")
    writeToFile(f"p-value for common words: {ttest(dist,varAcc2)}", "out.txt")
    writeToFile("","out.txt")
    graphPermRes(dist, [baseAcc, varAcc,varAcc2], ["baseline", "unique words","common words"])
