import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import joblib as jb 
import scipy as sp 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
#DataSet
from sklearn.datasets import load_iris

data=pd.read_csv("data.csv")

x=data['age'].values.reshape(-1,1)
y=data['ins'].values.reshape(-1,1).ravel()

x_train,x_test,y_train,y_test=train_test_split(x, y, random_state=0, test_size=0.2)

model=KNeighborsClassifier()
model.fit(x_train,y_train)

prediction=model.predict([[35]]) #Prediction
print(prediction)

