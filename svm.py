import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('sampleData.csv')
data.head()

data.shape

data.isnull().sum()

data.dropna()
"""
plt.figure(figsize=(8,8))
sns.countplot('TARGET',data=data)
plt.title(' balanced data or imbalance')
#plt.show()
"""
from sklearn.preprocessing import StandardScaler
data['NormalisAmount']=StandardScaler().fit_transform(data['REGION_RATING_CLIENT_W_CITY'].values.reshape(-1,1))
data.head()

data.drop(['REG_CITY_NOT_WORK_CITY','REGION_RATING_CLIENT_W_CITY'],axis=1,inplace=True) 

print(data.head())

shuffl_df = data.sample(frac=1,random_state=4)
shuffl_df.tail()

fraud=shuffl_df.loc[shuffl_df['TARGET']==1]
non_fraud=shuffl_df.loc[shuffl_df['TARGET']==0].sample(n=450,random_state=43)

data=pd.concat([fraud,non_fraud])
data = data.sample(frac=1,random_state=4)

"""
plt.figure(figsize=(8,8))
sns.countplot('TARGET',data=data)
plt.title(' balanced data or imbalance')
#plt.show()
"""

from sklearn.model_selection import train_test_split

x=data.iloc[:,data.columns!='TARGET']
y=data.iloc[:,data.columns=='TARGET']

xtrain, xtest, ytrain, ytest =train_test_split(x,y, test_size=0.3)

print(xtest.shape)

print(ytest.shape)

#svm
from sklearn import svm 
model1=svm.LinearSVC()
import warnings
warnings.filterwarnings('ignore')

model1.fit(xtrain,ytrain)

model1_predict=model1.predict(xtest)
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

acc1=accuracy_score(ytest,model1_predict)
print('SVM accuracy score:',acc1)

cnf_model1=confusion_matrix(ytest,model1_predict)
print(cnf_model1)  #TN #FP #FN #TP

classification_svm=classification_report(ytest,model1_predict)
print(classification_svm) #recall=TP/TP+FN


#finding correlation between columns and plotting heatmap

corrmat = data.corr() 
fig = plt.figure(figsize = (12, 9)) 
sns.heatmap(corrmat, vmax = .8, square = True) 
plt.show()


LABELS = ['Normal', 'Fraud'] 
conf_matrix = confusion_matrix(ytest,model1_predict) 
plt.figure(figsize =(12, 12)) 
sns.heatmap(conf_matrix, xticklabels = LABELS,  
            yticklabels = LABELS, annot = True, fmt ="d"); 
plt.title("Confusion matrix") 
plt.ylabel('True class') 
plt.xlabel('Predicted class') 
plt.show() 