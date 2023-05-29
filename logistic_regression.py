import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import joblib as jb 
import scipy as sp 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 



data=pd.read_csv("data.csv")
x=data[['independent_variables']].values
y=data[['dependent_variable']].values.ravel()

x_train,y_train,x_test,y_test=train_test_split(x,y,random_state=0,test_size=0.2)

model=LogisticRegression()

model_fit=model.fit(x_train,y_train)
prediction=model.predict(x_test)
print(prediction)



log_reg_model=jb.dump(model,"logistic_regression.h5") 

import joblib as jb

model=jb.load("logistic_regression.h5")
value=int(input())
pred=model.predict([[value]])
print(pred)

