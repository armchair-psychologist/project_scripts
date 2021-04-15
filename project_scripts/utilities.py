import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os
import plotly.express as px
import plotly
import seaborn as sns
import csv
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from bs4 import BeautifulSoup
from statsmodels.stats.weightstats import ttest_ind
import spacy
import unidecode 
import contractions
from nltk.stem import WordNetLemmatizer 
import re
nltk.download('wordnet')

'''
This will load the csv
'''

class CsvToDf:
    '''
    This class will simply turn the given data to a dataframe
    '''
    def __init__(self,filename,batchSize=None,cols=None,preProc=False,postCol=False,toReplace=None):
        #batchSize is the size of data to be read incrementally. This is for data that is to big to fit
        #into memory
        self._toReplace = toReplace
        self._preProc = preProc
        self._postCol = postCol
        self._cols = cols
        self._header = None
        self._filename = filename
        self._curIndex = 0     #this will be the current index that we are in the csv
        self._isRead = False
        self._df = None
        self._storeHeader()
        self._batchSize = batchSize
    def getBatchSize(self):
        return self._batchSize
    def preprocessing_v1(self,text):
        #remove html information
        soup = BeautifulSoup(text, "html.parser")
        processed = soup.get_text(separator=" ")

        #remove http// 
        processed = re.sub(r"http\S+", "", processed)

        #remove ||| seperate
        processed = re.sub(r'\|\|\|', r' ', processed)

        #lower case
        processed = processed.lower()

        #expand shortened words, e.g. don't to do not
        processed = contractions.fix(processed)

        #remove accented char
        processed = unidecode.unidecode(processed)

        #remove white space
        #processed = processed.strip()
        #processed = " ".join(processed.split())

        # Lemmatizing 
        lemmatizer = WordNetLemmatizer() 
        processed=lemmatizer.lemmatize(processed)
        return processed
    def _storeHeader(self):
        with open(self._filename) as csvFile:
            f = csv.reader(csvFile)
            self._header = next(f)
    def getWholeCsv(self):
        if not(self._isRead):
            if self._cols != None:
                self._df = pd.read_csv(self._filename,usecols=self._cols)
            else:
                self._df = pd.read_csv(self._filename)
            self._isRead = True
        return self._df
    def getHeader(self):
        return self._header
    def _checkIfRead(self):
        if not(self._isRead):
            if self._cols != None:
                self._df = pd.read_csv(self._filename,iterator=True,chunksize=self._batchSize,usecols=self._cols)
            else:
                self._df = pd.read_csv(self._filename,iterator=True,chunksize=self._batchSize)
            self._isRead = True
            return False
        return True
    def removeWord(self,text):
        words = text.split()
        cleanWords = [i for i in words if i not in self._toReplace]
        return " ".join(cleanWords)
    def getNextBatchCsv(self):
        self._checkIfRead()
        out = next(self._df,None)
        if self._preProc and isinstance(out,pd.DataFrame):
            out[self._postCol] = out[self._postCol].apply(self.preprocessing_v1)
        if self._toReplace != None and isinstance(out,pd.DataFrame):
            out[self._postCol] = out[self._postCol].apply(self.removeWord)
        return out

#================ counting the smallest number of data
TEST = "test"
TRAIN = "train"
class Combiner:
    '''
    - Given multiple CsvToDf that correspond to a dataset combine them to a single dataframe
    - return this dataframe
    - need to return a dataframe that only has type and post as its columns
    '''
    def __init__(self,dataList,columnList):
        '''
        dataList is the CsvToDf that contains all the data and columnList is a list that contains the necessary
        column names for a corresponding entry in dataList.
        '''
        assert len(dataList) == len(columnList),"incorrect sizes for data"
        self._dataList = dataList
        self._data = [None for i in range(len(dataList))]
        self._necessaryCol = columnList
        self._typeCol = "type"
        self._postCol = "posts"
        self._count = 0
        self._incrementData()
        
    def getCount(self):
        return self._count
    def getNextBatch(self):
        '''
        return a dataframe that contains all the aggregated data
        '''
        outData = pd.DataFrame(columns=[self._typeCol,self._postCol])
        for data,colList in zip(self._data,self._necessaryCol):
            if isinstance(data,pd.DataFrame):
                renamedData = data[[colList[0],colList[1]]]
                renamedData.columns = [self._typeCol,self._postCol]
                
                outData = outData.append(renamedData,ignore_index=True)
        self._incrementData()
        if (len(outData.index)) == 0:
            return None
        else:
            return outData
    def _incrementData(self):
        for idx,i in enumerate(self._dataList):
            self._count += i.getBatchSize()
            self._data[idx] = i.getNextBatchCsv()
class Balancer:
    '''
    - Balance the count
    - Decide what the training and test dat will be
    - Needs to output three data frames the train the test and the remainder
    - make the remainder the training set
    '''
    def __init__(self,combiner,trainFreq,testFreq):
        #personSize is minimum size of the number of people in a single personality group
        self._combiner = combiner
        self._typeCol = "type"
        self._postCol = "posts"
        self._personality_count = {"ENTJ" : {TRAIN:0,TEST:0}, "INTJ" : {TRAIN:0,TEST:0}, "ENTP" : {TRAIN:0,TEST:0}, "INTP" : {TRAIN:0,TEST:0}, "INFJ" : {TRAIN:0,TEST:0}, "INFP" : {TRAIN:0,TEST:0}, "ENFJ" : {TRAIN:0,TEST:0} , 
                    "ENFP" : {TRAIN:0,TEST:0}, "ESTP" : {TRAIN:0,TEST:0}, "ESTJ" : {TRAIN:0,TEST:0}, "ISTP" : {TRAIN:0,TEST:0}, "ISTJ" : {TRAIN:0,TEST:0}, "ISFJ" : {TRAIN:0,TEST:0}, "ISFP" : {TRAIN:0,TEST:0}, 
                    "ESFJ" : {TRAIN:0,TEST:0}, "ESFP" : {TRAIN:0,TEST:0}}
        self._trainFreq = trainFreq
        self._testFreq = testFreq
        self._training = []
        self._testing = []
    def createDataSets(self):
        self.reset()
        while not(self._trainIsUniform()) or not(self._testIsUniform()):
            #the three conditionals above will check if test and train dataset have uniform data 
            batch = self._combiner.getNextBatch()
            if not(isinstance(batch,pd.DataFrame)):
                break
            for idx,row in batch.iterrows():
                if isinstance(row[self._typeCol],str):
                    personality = row[self._typeCol].upper()
                    if personality in self._personality_count:
                        if self._personality_count[personality][TRAIN] < self._trainFreq:
                            self._training.append({self._typeCol:personality,self._postCol:row[self._postCol]})
                            self._personality_count[personality][TRAIN] += 1
                        elif self._personality_count[personality][TEST] < self._testFreq:
                            self._testing.append({self._typeCol:personality,self._postCol:row[self._postCol]})
                            self._personality_count[personality][TEST] += 1
        return True
    def reset(self):
        self._training = []
        self._testing = []
        self._personality_count = {"ENTJ" : {TRAIN:0,TEST:0}, "INTJ" : {TRAIN:0,TEST:0}, "ENTP" : {TRAIN:0,TEST:0}, "INTP" : {TRAIN:0,TEST:0}, "INFJ" : {TRAIN:0,TEST:0}, "INFP" : {TRAIN:0,TEST:0}, "ENFJ" : {TRAIN:0,TEST:0} , 
                    "ENFP" : {TRAIN:0,TEST:0}, "ESTP" : {TRAIN:0,TEST:0}, "ESTJ" : {TRAIN:0,TEST:0}, "ISTP" : {TRAIN:0,TEST:0}, "ISTJ" : {TRAIN:0,TEST:0}, "ISFJ" : {TRAIN:0,TEST:0}, "ISFP" : {TRAIN:0,TEST:0}, 
                    "ESFJ" : {TRAIN:0,TEST:0}, "ESFP" : {TRAIN:0,TEST:0}}
    def getCount(self):
        return self._combiner.getCount()
    def getTrainSet(self):
        return pd.DataFrame(self._training)
    def getTestSet(self):
        return pd.DataFrame(self._testing)
    def _trainIsUniform(self):
        #checks if personality count has equal distribution
        for key in self._personality_count:
            if self._personality_count[key][TRAIN] < self._trainFreq:
                return False
        return True
    def _testIsUniform(self):
        #checks if personality count has equal distribution
        for key in self._personality_count:
            if self._personality_count[key][TEST] < self._testFreq:
                return False
        return True
#======================================================