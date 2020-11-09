import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('sampleData.csv')
data.head()

data.shape

data.isnull().sum()

data.dropna()

plt.figure(figsize=(8,8))
sns.countplot('TARGET',data=data)
plt.title(' balanced data or imbalance')
plt.show()

from sklearn.preprocessing import StandardScaler
data['NormalisAmount']=StandardScaler().fit_transform(data['REGION_RATING_CLIENT_W_CITY'].values.reshape(-1,1))
data.head()

data.drop(['REG_CITY_NOT_WORK_CITY','REGION_RATING_CLIENT_W_CITY'],axis=1,inplace=True) 

data.head()

shuffl_df = data.sample(frac=1,random_state=4)
shuffl_df.tail()

fraud=shuffl_df.loc[shuffl_df['TARGET']==1]
non_fraud=shuffl_df.loc[shuffl_df['TARGET']==0].sample(n=450,random_state=43)

data=pd.concat([fraud,non_fraud])
data = data.sample(frac=1,random_state=4)

plt.figure(figsize=(8,8))
sns.countplot('TARGET',data=data)
plt.title(' balanced data or imbalance')
plt.show()

from sklearn.model_selection import train_test_split

x=data.iloc[:,data.columns!='TARGET']
y=data.iloc[:,data.columns=='TARGET']

xtrain, xtest, ytrain, ytest =train_test_split(x,y, test_size=0.3)

print(xtest.shape)

print(ytest.shape)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

model2=RandomForestClassifier()

model2.fit(xtrain,ytrain)
model2_predict=model2.predict(xtest)

acc2=accuracy_score(ytest,model2_predict)
print('Random Forest accuracy score:',acc2)
