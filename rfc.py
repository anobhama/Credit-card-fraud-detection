import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from matplotlib import gridspec 

data = pd.read_csv("sampleData.csv") 

print(data.head())

print(data.shape) 
print(data.describe()) 

fraud = data[data['TARGET'] == 1] 
valid = data[data['TARGET'] == 0] 
outlierFraction = len(fraud)/len(valid)
print(outlierFraction) 
print('Fraud Cases: {}'.format(len(data[data['TARGET'] == 1]))) 
print('Valid Transactions: {}'.format(len(data[data['TARGET'] == 0])))

#print(“details of the fraudulent transaction”) 
fraud.REGION_RATING_CLIENT_W_CITY.describe() 

valid.REGION_RATING_CLIENT_W_CITY.describe() 
"""
data.drop('NAME_CONTRACT_TYPE',axis='columns', inplace=True)
print(data.head())
"""

# dividing the X and the Y from the dataset 
X = data.drop(['TARGET'], axis = 1) 
Y = data["TARGET"] 
print(X.shape) 
print(Y.shape) 

# getting just the values for the sake of processing  
# (its a numpy array with no columns) 
xData = X.values 
yData = Y.values 

# Using Skicit-learn to split data into training and testing sets 
from sklearn.model_selection import train_test_split 
# Split the data into training and testing sets 
xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size = 0.2, random_state = 42) 

# Building the Random Forest Classifier (RANDOM FOREST) 
from sklearn.ensemble import RandomForestClassifier 
# random forest model creation 
rfc = RandomForestClassifier() 
rfc.fit(xTrain, yTrain) 
# predictions 
yPred = rfc.predict(xTest)

from sklearn.metrics import classification_report, accuracy_score  
from sklearn.metrics import precision_score, recall_score 
from sklearn.metrics import f1_score, matthews_corrcoef 
from sklearn.metrics import confusion_matrix 
  
n_outliers = len(fraud) 
n_errors = (yPred != yTest).sum() 
print("The model used is Random Forest classifier") 
  
acc = accuracy_score(yTest, yPred) 
print("The accuracy is {}".format(acc)) 
  
prec = precision_score(yTest, yPred) 
print("The precision is {}".format(prec)) 
  
rec = recall_score(yTest, yPred) 
print("The recall is {}".format(rec)) 
  
f1 = f1_score(yTest, yPred) 
print("The F1-Score is {}".format(f1)) 

MCC = matthews_corrcoef(yTest, yPred) 
print("The Matthews correlation coefficient is{}".format(MCC)) 