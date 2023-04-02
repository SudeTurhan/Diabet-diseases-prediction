# Diabet-diseases-prediction
Overview:

The dataset used in this project is from the National Institute of Diabetes and Digestive and Kidney Diseases, and the objective is to predict whether a patient has diabetes based on certain diagnostic measurements. The attributes of the dataset include pregnancies, glucose level, blood pressure, skin thickness, insulin level, BMI, diabetes pedigree function, age, and outcome (1 for yes and 0 for no).

CRISP-DM Data Science Project Management Method:

The CRISP-DM method was used for this project, and classification was used to determine whether the patient has diabetes or not. The dataset was checked for missing data, and none was found. The features and columns were separated, and the dataset was split into training and testing sets to avoid overfitting.

Feature Scaling:

Feature scaling was applied to compress the data into a narrower range when the difference between numerical data was too great. Standard Scaler was used to scale the data, which translates variables into a distribution with a mean of 0 and a standard deviation of 1. This method improves the performance of some models, such as distance-based algorithms (KNN, SVM), and the speed of algorithms using gradient descent.

Dimension Transformation:

PCA is an unsupervised dimensionality reduction technique used in clustering problems, while LDA is a supervised technique used in classification problems. LDA aims to find dimensions that maximize the separation between classes. Both methods reduce the dimensionality of the dataset, preventing overfitting and reducing computational costs.

K-Nearest Neighbor:

The KNN algorithm uses known training data to find the k points closest to a new data point and accepts the label of the majority as the label of the new point. The k value that gives the best accuracy score was obtained with a for loop, and the KNN algorithm was applied before and after the preprocessing methods to compare the results.

Decision Tree Classifier:

The decision tree classifier uses a structure to divide a dataset into smaller sets by applying a set of decision rules. The max_depth value that gives the best accuracy score was obtained with a for loop, and the decision tree algorithm was applied before and after the preprocessing methods to compare the results.


Results:

The accuracy score and confusion matrix were used for evaluation. The train and predict times of the models were measured and compared. The KNN and DTC algorithms were run with and without preprocessing to observe the effect of the methods. The accuracy scores did not increase as expected, and it was observed that the scores decreased after preprocessing. This may be due to the small number of features and rows in the dataset. 


















