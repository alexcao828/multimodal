import pandas as pd
import os
import shutil
import numpy as np

os.mkdir('imgs_PhaseClassFoldered')

train = pd.read_csv('train.csv')
train['wordsRoots'] = train['words'].apply(lambda x: x[0:-9])
train['labels'] = np.where(train['roots'] == train['wordsRoots'], 'match', 'notMatch')
os.mkdir('imgs_PhaseClassFoldered/train')
labels = np.unique(train['labels'])
for i in range(len(labels)):
    this_label = train[train['labels'].isin([labels[i]])].reset_index(drop=True)
    os.mkdir('imgs_PhaseClassFoldered/train/'+labels[i])
    for j in range(this_label.shape[0]):
        shutil.copyfile('thumbnails/'+this_label['images'][j], 'imgs_PhaseClassFoldered/train/'+labels[i]+'/'+this_label['images'][j])

val = pd.read_csv('val.csv')
val['wordsRoots'] = val['words'].apply(lambda x: x[0:-9])
val['labels'] = np.where(val['roots'] == val['wordsRoots'], 'match', 'notMatch')
os.mkdir('imgs_PhaseClassFoldered/val')
labels = np.unique(val['labels'])
for i in range(len(labels)):
    this_label = val[val['labels'].isin([labels[i]])].reset_index(drop=True)
    os.mkdir('imgs_PhaseClassFoldered/val/'+labels[i])
    for j in range(this_label.shape[0]):
        shutil.copyfile('thumbnails/'+this_label['images'][j], 'imgs_PhaseClassFoldered/val/'+labels[i]+'/'+this_label['images'][j])

test = pd.read_csv('test.csv')
test['wordsRoots'] = test['words'].apply(lambda x: x[0:-9])
test['labels'] = np.where(test['roots'] == test['wordsRoots'], 'match', 'notMatch')
os.mkdir('imgs_PhaseClassFoldered/test')
labels = np.unique(test['labels'])
for i in range(len(labels)):
    this_label = test[test['labels'].isin([labels[i]])].reset_index(drop=True)
    os.mkdir('imgs_PhaseClassFoldered/test/'+labels[i])
    for j in range(this_label.shape[0]):
        shutil.copyfile('thumbnails/'+this_label['images'][j], 'imgs_PhaseClassFoldered/test/'+labels[i]+'/'+this_label['images'][j])
        