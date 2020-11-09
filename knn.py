import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


data = pd.read_csv("sampleData.csv")

data.shape

data.head()

print(data["TARGET"].value_counts())

#taking first 20000 samples
data_20000 = data[:35000]

print(data_20000.shape)

print(data_20000["TARGET"].value_counts())
data20000 = data_20000.drop(['TARGET'], axis=1)
data20000.shape

data20000_labels = data_20000["TARGET"]
data20000_labels.shape

data20000_Std = StandardScaler().fit_transform(data20000)
print(data20000_Std.shape)
print(type(data20000_Std))

#KNN on data set - 5 fold CV


X1 = data20000_Std[0:27000]
XTest = data20000_Std[27000:35000]
Y1 = data20000_labels[0:27000]
YTest = data20000_labels[27000:35000]

myList = list(range(0,50))
neighbors = list(filter(lambda x: x%2!=0, myList))  #This will give a list of odd numbers only ranging from 0 to 50

CV_Scores = []

for k in neighbors:
    KNN = KNeighborsClassifier(n_neighbors = k, algorithm = 'kd_tree')
    scores = cross_val_score(KNN, X1,Y1, cv = 5, scoring='recall')
    CV_Scores.append(scores.mean())

print(CV_Scores)

plt.figure(figsize = (14, 12))
plt.plot(neighbors, CV_Scores)
plt.title("Neighbors Vs Recall Score", fontsize=25)
plt.xlabel("Number of Neighbors", fontsize=25)
plt.ylabel("Recall Score", fontsize=25)
plt.grid(linestyle='-', linewidth=0.5)
#plt.show()

best_k = neighbors[CV_Scores.index(max(CV_Scores))]
print(best_k)


#print(YTest.value_counts())

from sklearn.metrics import recall_score

KNN_best = KNeighborsClassifier(n_neighbors = best_k, algorithm = 'kd_tree')

KNN_best.fit(X1, Y1)

prediction = KNN_best.predict(XTest)

recallTest = recall_score(YTest, prediction)

print("Recall Score of the knn classifier for best k values of "+str(best_k)+" is: "+str(recallTest))

cm = confusion_matrix(YTest, prediction)

print(cm)

tn, fp, fn, tp = cm.ravel()

print(tn, fp, fn, tp)

print(YTest.value_counts())


# Calculating R square value of our model
from sklearn.metrics import r2_score

print("Recall Score of the knn classifier for best k values of "+str(best_k)+" is: "+str(recallTest))
