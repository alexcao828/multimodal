import pandas as pd
import os
import shutil
import numpy as np

os.mkdir('imgs_PhaseClassFoldered')

train = pd.read_csv('train.csv')
os.mkdir('imgs_PhaseClassFoldered/train')
classes = np.unique(train['section'])
for i in range(len(classes)):
    this_class = train[train['section'].isin([classes[i]])].reset_index(drop=True)
    os.mkdir('imgs_PhaseClassFoldered/train/'+classes[i])
    for j in range(this_class.shape[0]):
        shutil.copyfile('imgs/'+this_class['image_id'][j]+'.jpg', 'imgs_PhaseClassFoldered/train/'+classes[i]+'/'+this_class['image_id'][j]+'.jpg')

val = pd.read_csv('val.csv')
os.mkdir('imgs_PhaseClassFoldered/val')
classes = np.unique(val['section'])
for i in range(len(classes)):
    this_class = val[val['section'].isin([classes[i]])].reset_index(drop=True)
    os.mkdir('imgs_PhaseClassFoldered/val/'+classes[i])
    for j in range(this_class.shape[0]):
        shutil.copyfile('imgs/'+this_class['image_id'][j]+'.jpg', 'imgs_PhaseClassFoldered/val/'+classes[i]+'/'+this_class['image_id'][j]+'.jpg')

test = pd.read_csv('test.csv')
os.mkdir('imgs_PhaseClassFoldered/test')
classes = np.unique(test['section'])
for i in range(len(classes)):
    this_class = test[test['section'].isin([classes[i]])].reset_index(drop=True)
    os.mkdir('imgs_PhaseClassFoldered/test/'+classes[i])
    for j in range(this_class.shape[0]):
        shutil.copyfile('imgs/'+this_class['image_id'][j]+'.jpg', 'imgs_PhaseClassFoldered/test/'+classes[i]+'/'+this_class['image_id'][j]+'.jpg')

