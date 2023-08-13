import pandas as pd
import numpy as np

embeddings_dict = {}
with open("glove.6B/glove.6B.300d.txt", 'r') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector

root = '/esp/data/challenges-in-representation-learning-multi-modal-learning/ESPGame100k/labels/'

words = []
train = pd.read_csv('train.csv')
for i in range(train.shape[0]):
    file = root+train.words[i]
    with open(file, 'rb') as f:
        lines = f.readlines()
    lines = [x.decode('utf-8')[0:-1] for x in lines if (x.decode('utf-8')[0:-1] in embeddings_dict)]
    words.extend(lines)
val = pd.read_csv('val.csv')
for i in range(val.shape[0]):
    file = root+val.words[i]
    with open(file, 'rb') as f:
        lines = f.readlines()
    lines = [x.decode('utf-8')[0:-1] for x in lines if (x.decode('utf-8')[0:-1] in embeddings_dict)]
    words.extend(lines)
test = pd.read_csv('test.csv')
for i in range(test.shape[0]):
    file = root+test.words[i]
    with open(file, 'rb') as f:
        lines = f.readlines()
    lines = [x.decode('utf-8')[0:-1] for x in lines if (x.decode('utf-8')[0:-1] in embeddings_dict)]
    words.extend(lines)
values, counts = np.unique(words, return_counts=True)
countDict = dict(zip(values, -counts))

numWords = 42
gloveDim = 300

trainGlove = np.zeros((train.shape[0], numWords, gloveDim))
for i in range(train.shape[0]):
    file = root+train.words[i]
    with open(file, 'rb') as f:
        lines = f.readlines()
    lines = [x.decode('utf-8')[0:-1] for x in lines if (x.decode('utf-8')[0:-1] in embeddings_dict)]
    sortedWords = sorted(lines, key=countDict.get)
    for j in range(len(sortedWords)):
        trainGlove[i, j, :] = embeddings_dict[sortedWords[j]]
np.save('trainGlove.npy', trainGlove)
        
valGlove = np.zeros((val.shape[0], numWords, gloveDim))
for i in range(val.shape[0]):
    file = root+val.words[i]
    with open(file, 'rb') as f:
        lines = f.readlines()
    lines = [x.decode('utf-8')[0:-1] for x in lines if (x.decode('utf-8')[0:-1] in embeddings_dict)]
    sortedWords = sorted(lines, key=countDict.get)
    for j in range(len(sortedWords)):
        valGlove[i, j, :] = embeddings_dict[sortedWords[j]]
np.save('valGlove.npy', valGlove)   
        
testGlove = np.zeros((test.shape[0], numWords, gloveDim))
for i in range(test.shape[0]):
    file = root+test.words[i]
    with open(file, 'rb') as f:
        lines = f.readlines()
    lines = [x.decode('utf-8')[0:-1] for x in lines if (x.decode('utf-8')[0:-1] in embeddings_dict)]
    sortedWords = sorted(lines, key=countDict.get)
    for j in range(len(sortedWords)):
        testGlove[i, j, :] = embeddings_dict[sortedWords[j]]
np.save('testGlove.npy', testGlove)    
        