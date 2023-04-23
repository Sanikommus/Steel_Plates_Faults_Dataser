import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# Part A
#Reading csv file 
df = pd.read_csv('SteelPlateFaults-2class.csv')

#splitting the data in 70 and 30
[data_train , data_test] = train_test_split(df , test_size = 0.3 , random_state = 42 , shuffle = True)

#dropping the unnecessary columns
df.drop(columns=['X_Minimum','Y_Minimum','TypeOfSteel_A300','TypeOfSteel_A400'], axis=1)
data_train = data_train.drop(columns=['X_Minimum','Y_Minimum','TypeOfSteel_A300','TypeOfSteel_A400'], axis=1)
data_test = data_test.drop(columns=['X_Minimum','Y_Minimum','TypeOfSteel_A300','TypeOfSteel_A400'], axis=1)

#grouping the classes
by_class = data_train.groupby('Class')

data_train_0 = by_class.get_group(0)
data_train_1 = by_class.get_group(1)

#getting the x_train and x_label_yrain for various classes
data_train_c0 = data_train_0['Class']
data_train_0 = data_train_0.drop(['Class'], axis=1)

data_train_c1 = data_train_1['Class']
data_train_1 = data_train_1.drop(['Class'], axis=1)

#getting the testing data into x_test and x_label_test for various classes
data_test_c = data_test['Class']
data_test = data_test.drop(['Class'], axis= 1)
Q=[2,4,8,16]


acc=[]


for q in Q :
    
    #defining the multivariate gaussian model and training it for various classes
    c0 = GaussianMixture(n_components=q , covariance_type='full' )
    c0 .fit(data_train_0.values)
    
    c1 = GaussianMixture(n_components=q , covariance_type='full' )
    c1.fit(data_train_1.values)
    pred = []

    #calculating the score
    a = c0.score_samples(data_test.values)
    b = c1.score_samples(data_test.values)
    
    #assigning label to the test data based on the probability
    for i in range (len(a)):
        if a[i] > b[i] :
            pred.append(0)
        if a[i] < b[i] :
            pred.append(1)

    #calculating the accuracy score and the confusion matrix
    matrix = confusion_matrix (data_test_c.values, pred)
    accuracy = accuracy_score(data_test_c.values, pred)
    acc.append(accuracy)
    
    print("confusion matrix for q = " , q, "is\n" , matrix)
    print("accuracy score for q = ",q, "is :" , accuracy.round(3))

#looking out and printing the value Q for which accuracy is max 
for i in range(len(Q)):
    if acc[i]==max(acc):
        Qmax=Q[i]
print("Accuracy will be maximum for Q =",Qmax)
