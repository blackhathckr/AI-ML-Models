import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import joblib as jb 
import scipy as sp 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data=pd.read_csv("data.csv")

encoder=LabelEncoder() 
x=data[['company','job','degree']]
y=data['sal']

x['company']=encoder.fit_transform(x['company'])
x['job']=encoder.fit_transform(x['job'])
x['degree']=encoder.fit_transform(x['degree'])

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

model=DecisionTreeClassifier()
model.fit(x_train,y_train)

prediction=model.predict([[2,0,1]])

print(prediction)