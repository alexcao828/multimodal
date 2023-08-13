import pandas as pd
import os
from sklearn.utils import shuffle

nytimes = pd.read_json('N24News/news/nytimes.json')

imgs = os.listdir('N24News/imgs')
imgs = [img[0:-4] for img in imgs]

nytimes = nytimes[nytimes['image_id'].isin(imgs)]

nytimes = shuffle(nytimes).reset_index(drop=True)

M = nytimes.shape[0]
M = int(M / 10)

test = nytimes.iloc[0:M, :]
val = nytimes.iloc[M:2*M, :]
train = nytimes.iloc[2*M:, :]

test.to_csv('test.csv', index=False)
val.to_csv('val.csv', index=False)
train.to_csv('train.csv', index=False)
