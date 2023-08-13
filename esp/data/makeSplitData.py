import os
import pandas as pd
from sklearn.utils import shuffle
import numpy as np

words_files = os.listdir('challenges-in-representation-learning-multi-modal-learning/ESPGame100k/labels')
words_roots = [words_file[0:-9] for words_file in words_files]
words = pd.DataFrame(
    {'roots': words_roots,
     'words': words_files
    })

images_files = os.listdir('challenges-in-representation-learning-multi-modal-learning/ESPGame100k/thumbnails')
images_roots = [images_file[0:-4] for images_file in images_files]
images = pd.DataFrame(
    {'roots': images_roots,
     'images': images_files
    })

both = pd.merge(images, words, how="inner", on=["roots"])
both = shuffle(both).reset_index(drop=True)

M = int(both.shape[0]/2)
match = both.iloc[0:M, :]
notMatch = both.iloc[M:, :]

notMatch.loc[:, 'words'] = np.roll(notMatch.words, 1)

M = int(match.shape[0]/10)
matchTrain =  match.iloc[0:8*M, :]
notMatchTrain =  notMatch.iloc[0:8*M, :]
train = pd.concat([matchTrain, notMatchTrain])
train = shuffle(train).reset_index(drop=True)
train.to_csv('train.csv', index=False)

matchVal =  match.iloc[8*M:9*M, :]
notMatchVal =  notMatch.iloc[8*M:9*M, :]
val = pd.concat([matchVal, notMatchVal])
val = shuffle(val).reset_index(drop=True)
val.to_csv('val.csv', index=False)

matchTest =  match.iloc[9*M:, :]
notMatchTest =  notMatch.iloc[9*M:, :]
test = pd.concat([matchTest, notMatchTest])
test = shuffle(test).reset_index(drop=True)
test.to_csv('test.csv', index=False)
