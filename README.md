# Steel_Plates_Faults_Dataset

Given the Steel Plates Faults Data Set as a csv file (SteelPlateFaults2class.csv). This dataset comes from research by Semeion, Research Center of Sciences of
Communication. The original aim of the research was to correctly classify the type of surface
defects in stainless steel plates, with six types of possible defects (plus "other"). The Input vector is
made up of 27 indicators that approximately describe the geometric shape of the defect and its
outline. As Semeion was commissioned by the Centro Sviluppo Materiali (Italy) for this task and
therefore the details of the nature of the 27 indicators used as input vectors or the types of the 6
classes of defects are confidential.

The dataset used for this assignment contains features extracted from the steel plates of types A300
and A400 to predict whether the image of the surface of steel plate contains two types of faults such
as Z_Scratch and K-Scratch. It consists 1119 tuples each having 27 attributes which are indicators
representing the geometric shape of the fault. The last attribute (28th attribute) for every tuple
signifies the class label (0 for K_Scratch fault and 1 for Z_Scratch fault). It is a two-class problem.


The code:

* Imports of KNN, Test_train_split, confusion_matrix and accuracy from sklearn.
* Loads the data and then splits its into training and testing dataset.
* Trains the KNN model on the training data and then checks the accuracy on the testing data for various values of K.
* Agains does the same thing but on the normalized data.
* Trains a Bayes Classifier model on the training data and also calculates the accuacy on the testing data.
* Also, prints the confusion matrix for various madels and its parameters.
* Trains a GMM model with different values of Q and calculates the confusion matrix and accuracy.


# Input Dataset

https://www.kaggle.com/datasets/uciml/faulty-steel-plates

![image](https://user-images.githubusercontent.com/119813195/228888338-b5597948-fe46-4eb3-966a-84ab47247832.png)

# Output 

KNN without normalization of dataset:

![image](https://user-images.githubusercontent.com/119813195/228889159-88e28924-fe7c-4e09-bb17-d9dc694b1c59.png)

KNN with normalization of dataset:

![image](https://user-images.githubusercontent.com/119813195/228889459-f83ceb5d-7bdd-4c7f-bbf2-2d5a6a47a2ab.png)
