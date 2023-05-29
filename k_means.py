import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import joblib as jb 
import scipy as sp 
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split 

data=pd.read_csv("data.csv")

x=data['independent_variables'].values.reshape(-1,1)
y=data['dependent_variable'].values.reshape(-1,1)



model=KMeans(n_clusters=3)

pred=model.fit_predict(x,y)

data['cluster']=pred 

df1=data[data.cluster==0]
df2=data[data.cluster==1]
df3=data[data.cluster==2]

plt.scatter(df1['age'],df1['income'],color='blue')
plt.scatter(df2['age'],df2['income'],color='green')
plt.scatter(df3['age'],df3['income'],color='red')
plt.show()

print(model.predict([[45]])) # Prediction
