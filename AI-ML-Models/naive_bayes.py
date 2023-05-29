import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import joblib as jb 
import scipy as sp 
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split

data=pd.read_csv("data.csv")

x=data['independent_variables'].values.reshape(-1,1)
y=data['dependnt_variable'].values.reshape(-1,1)

x_train,x_test,y_train,y_test=train_test_split(x,y)

model=BernoulliNB()
model.fit(x_train,y_train)
prediction=model.predict([[32]]) # Prediction
print(x_test)
print(int(prediction))