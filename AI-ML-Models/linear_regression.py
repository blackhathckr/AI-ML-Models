import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import joblib as jb 
import scipy as sp 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 


data=pd.read_csv("data.csv")
x=data['independent_variables'].values.reshape(-1,1)
y=data['dependent_variable'].values

x_train,y_train,x_test,y_test=train_test_split(x,y,random_state=0,test_size=0.2)

model=LinearRegression()

model_fit=model.fit(x_train,y_train)
prediction=model.predict(x_test)
print(prediction)


# Saving/Dumping the ML model using joblib

lin_reg_model=jb.dump(model,"linear_regression.h5") 


# Loading the ML model using joblib

import joblib as jb

model=jb.load("linear_regression.h5")
value=int(input())
pred=model.predict([[value]])
print(pred) 

