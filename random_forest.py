import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import joblib as jb 
import scipy as sp 
from sklearn.ensemble import RandomForestClassifier

#DataSet

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import random

digits=load_digits()

data=pd.DataFrame(digits.data)
data['target']=digits.target

x=data.drop(['target'],axis='columns')
y=data['target']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

model=RandomForestClassifier()
model.fit(x_train,y_train)

user=[]
for i in range(64):
    val=random.randint(0,15)
    user.append(val)

plt.imshow([user])
plt.show()

prediction=model.predict([user])

print(prediction)
