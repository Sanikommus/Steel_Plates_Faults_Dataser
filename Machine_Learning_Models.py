# Importing required libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Question 1
# Reading the csv file
df = pd.read_csv("C:/Users/manta/OneDrive/Desktop/Semester/Semester 3/DATA SCIENCE 3/B21015_Lab_5_Ds3/SteelPlateFaults-2class.csv")

# Creating test and train dataframe
[X_train , X_test] = train_test_split(df , test_size = 0.3 , random_state = 42 , shuffle = True)

X_train_ex = X_train.copy()
X_test_ex = X_test.copy()

# Converting train test dataframe into csv
X_train.to_csv("SteelPlateFaults-train.csv")
X_test.to_csv("SteelPlateFaultstest.csv")

# Generating xtraining and ytraining samples
X_label_train = X_train["Class"]
X_train = X_train.drop(["Class"] , axis = "columns" )

# Generating xtesting samples and ytraining samples
X_label_test = X_test["Class"]
X_test = X_test.drop(["Class"] , axis = "columns" )


#Q1 Part A
score = []
cmatrix = []
accuracy = {}

for i in [1,3,5]:
    knn = KNeighborsClassifier(n_neighbors = i )
    knn.fit(X_train , X_label_train)
    

    
    Y_pred = knn.predict(X_test)
    
    accuracy[i] = accuracy_score(X_label_test , Y_pred)
    cmatrix.append((confusion_matrix(X_label_test , Y_pred)))
        
q = sorted(accuracy.values())[::-1]

print("The confusion matrix :")

print("K=1 :" )
print(cmatrix[0])
print("K=3 :" )
print(cmatrix[1])
print("K=5 :" )
print(cmatrix[2])

for i in accuracy:
    if(accuracy[i] == q[0]):
        print("The highest classification accuracy percentage is for K value = " + str(i))

print("\n\n")

#Q1 Part B
X_train1 = X_train.copy()
X_test1 = X_test.copy()

#storing the column name in the list col
col = X_train.columns

#Normalizing the data using max and min values
for i in col:
    X_train1[i] = (X_train1[i] - min(X_train1[i])) / (max(X_train1[i]) - min(X_train1[i]))
    X_test1[i] = ((X_test1[i] - min(X_train1[i])) / (max(X_train1[i]) - min(X_train1[i])))

#concatinting to x and y for training and testing data    
s1 = pd.concat([X_train1 , X_label_train] , axis = 1)
s2 = pd.concat([X_test1 , X_label_test] , axis = 1)

s1.to_csv("SteelPlateFaults-train-Normalised.csv")
s2.to_csv("SteelPlateFaults-test-normalised.csv")

score1 = []
cmatrix1 = []
accuracy1 = {}

for i in [1,3,5]:
    knn1 = KNeighborsClassifier(n_neighbors = i )
    knn1.fit(X_train1 , X_label_train)
    
    score1.append(knn1.score(X_test1 , X_label_test))
    
    Y_pred1 = knn1.predict(X_test1)
    
    accuracy1[i] = accuracy_score(X_label_test , Y_pred1)
    cmatrix1.append((confusion_matrix(X_label_test , Y_pred1)))
        
q1 = sorted(accuracy1.values())[::-1]

print("The confusion matrix :")

print("K=1 :" )
print(cmatrix1[0])
print("K=3 :" )
print(cmatrix1[1])
print("K=5 :" )
print(cmatrix1[2])

for i in accuracy1:
    if(accuracy1[i] == q1[0]):
        print("The highest classification accuracy percentage is for K value = " + str(i))
        break;
        
print("\n\n")      

# Question 3
# Making required train and test dataframe corresponding to class 0 and 1.
F_train0_df = X_train_ex[X_train_ex["Class"]==0]
F_train1_df = X_train_ex[X_train_ex["Class"]==1]
F_train0_df = F_train0_df.drop(['X_Minimum', 'Y_Minimum', 'TypeOfSteel_A300', 'TypeOfSteel_A400',"Class"],axis = 1)
F_train1_df = F_train1_df.drop(['X_Minimum', 'Y_Minimum', 'TypeOfSteel_A300', 'TypeOfSteel_A400',"Class"],axis = 1)

# Finding prior probability,mean vector, covariance matrix  for class 0
Prior_prob0 = len(F_train0_df.index)/len(X_train_ex.index)
mean0 = round(F_train0_df.mean(),3)
cov0 = round(F_train0_df.T.cov(),3)
# Finding prior probability,mean vector, covariance matrix  for class 1
Prior_prob1 = len(F_train1_df.index)/len(X_train_ex.index)
mean1 = round(F_train1_df.mean(),3)
cov1 = round(F_train1_df.T.cov(),3)
cov0 = np.cov(F_train0_df.T)
cov1 = np.cov(F_train1_df.T)
# Defining testing data.
x_test = X_test_ex.drop(['X_Minimum', 'Y_Minimum', 'TypeOfSteel_A300', 'TypeOfSteel_A400',"Class"],axis = 1)
y_testb = X_label_test
x_testb = x_test 
# Intialising empty list for predicted class.
y_predb = []

# Taking each row of test data and classifying it according to bayes classifier.
for i in range(len(x_testb)):
  # Test vectors for class 0 and 1
  x0 = np.array(round(x_testb.iloc[i]-np.array(mean0),3))
  x1 = np.array(round(x_testb.iloc[i]-np.array(mean1),3))
  # Computing likelihood for class 0
  c0a= np.dot(np.transpose(x0),np.linalg.inv(cov0))
  c0b = np.dot(c0a, (x0))
  # Calculating likelihood for class 0.
  lik0 =  (np.exp((-0.5 * c0b))/ (((2 * np.pi)**11.5) * (np.linalg.det(cov0)**0.5)))
  
  # Computing likelihood for class 1
  c1a= np.dot(np.transpose(x1), np.linalg.inv(cov1))
  c1b = (np.dot(c1a, (x1)))
  # Calculating likelihood for class 1
  lik1 =  ((np.exp(-0.5 * c1b)) / (((2 * np.pi)**11.5) * (np.linalg.det(cov1)**0.5)))
  
  # Finding total probability such that x belongs to data.
 
  # Calculting posterior probability that x belongds to class 1 or class 0
  postprob0 = (lik0*Prior_prob0)/(lik1*Prior_prob1 + lik0*Prior_prob0)
  postprob1 = (lik1*Prior_prob1)/(lik1*Prior_prob1 + lik0*Prior_prob0)
  
  # Classifying according to posterior probability.
  if(postprob0>postprob1):
    y_predb.append(0)    
  else:
    y_predb.append(1)

y_predb = np.array(y_predb)

print("The confusion matrix of bayes classification is : ")
print((confusion_matrix(y_testb, y_predb)))

print("The  classification accuracy percentage for this bayes classification  is "+str(accuracy_score(y_testb, y_predb)*100))
a = accuracy_score(y_testb, y_predb)*100

# Building a data frame for comparing result of knn , knn with normalised data, bayes classifier.
d =[accuracy[5],accuracy1[5],a]
index=['knn','knn with normalized data','bayes classifier']
df=pd.DataFrame(d,index=index)
df.columns=['Accuracy in percentage']
fig = plt.figure(figsize = (8, 2))
ax = fig.add_subplot(111)

ax.table(cellText = df.values,
          rowLabels = df.index,
          colLabels = df.columns,
          loc = "center"
         )
ax.set_title("Comparison of classification accuracy")

ax.axis("off");
print("\n\n\n\n\n")
