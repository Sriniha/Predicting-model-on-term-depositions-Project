# DataScience-Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.linear_model import LogisticRegression

df=pd.read_csv(r"C:\Users\91709\Downloads\prob_csv\train.csv")

df1=pd.read_csv(r"C:\Users\91709\Downloads\prob_csv\test.csv")

df.shape

df1.shape

df.head()

train=df

test=df1

x_train=train.drop('subscribed',axis=1)

y_train=train['subscribed']

x_train=pd.get_dummies(x_train)

logreg=LogisticRegression()

logreg.fit(x_train,y_train)

x_test=pd.get_dummies(test)

pred=logreg.predict(x_test)

pred

logreg.score(x_train,y_train)

pred

from sklearn.metrics import accuracy_score

submission=pd.DataFrame()

submission['ID']=test['ID']
submission['subscribed']=pred

submission['subscribed'].replace(0,'no',inplace=True)
submission['subscribed'].replace(1,'yes',inplace=True)

submission.to_csv('submission.csv', header=True, index=False)

df3=pd.read_csv(r"submission.csv")

df3['subscribed']

