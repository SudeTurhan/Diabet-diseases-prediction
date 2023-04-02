# Diabet-diseases-prediction
Overview

Dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether a patient has diabetes, based on certain diagnostic measurements included in the dataset. 

Information about dataset attributes :

Pregnancies: To express the Number of pregnancies
Glucose: To express the Glucose level in blood
BloodPressure: To express the Blood pressure measurement
SkinThickness: To express the thickness of the skin
Insulin: To express the Insulin level in blood
BMI: To express the Body mass index
DiabetesPedigreeFunction: To express the Diabetes percentage
Age: To express the age
Outcome: To express the final result 1 is Yes and 0 is No



I use CRISP DM data science project management method. I used classification to find out if the patient has diabetes or not. First of all, I checked whether my data set also has missing data. There is no missing data.  Then I separated feature and columns. I split the dataset into training and testing to avoid overfitting.

Feature Scaling

While preparing the data for the model, the first thing I do on the data is feature scaling. The purpose of the method is to compress the data into a narrower range in cases where the difference between the numerical data is too great. I used feature scaling as some models have better performance and better results with data scaling. In particular, it affects the result of distance-based algorithms (KNN, SVM), while it affects the speed of algorithms using gradient descent. Standard Scaler, a Feature Scaling type, translates variables into a distribution with a mean of 0 std deviation of 1. It is found by subtracting the corresponding column average from all the data in the data set and dividing it by the column std deviation. Thus, all observation units in the data set get values between -1 and 1.
      

Dimension Transformation

PCA is an unsupervised while LDA is a supervised dimensionality reduction technique. PCA is used in clustering problems, while LDA is used in classification problems.
LDA aims to find dimensions that maximize separation between classes. Reduces dimension in the dataset. The aim is to prevent overfitting and at the same time reduce computational costs. 
       

K-Nearest Neighbor

KNN algorithm tags take known training data and keep it aside. When it sees a new data point, it returns to that data and finds the k points "closest" to that point. Then it looks at the labels of these points and accepts the label of the majority as the label of the new point. With the for loop, I got the k value that gives the best accuracy score from the k values. To see the effect of the preprocessing methods, I applied the KNN algorithm a total of two times before and after applying the preprocessing methods and compared the results.



Decision Tree Clasifier

A decision tree is a structure used to divide a dataset containing a large number of records into smaller sets by applying a set of decision rules. In other words, it is a structure used by dividing large amounts of records into very small record groups by applying simple decision-making steps. With the for loop, I got the max_depth value that gives the best accuracy score from the max_depth values. To see the effect of the preprocessing methods, I applied the Decision Tree algorithm a total of two times before and after applying the preprocessing methods and compared the results. Accordingly, I observed that the accuracy score of the tree decreased after preprocessing.



Result 

I used acc and confusion matrix for evaluation. I measured the train and predict times of the models. I then compared these results. To see the effect of the preprocessing methods on the algorithm, I ran the KNN and DTC algorithms without and after the preprocessing methods. I was expecting the accuracy scores to increase. But I have observed that the results of KNN acc (without preprocessing) and DTC acc (with preprocessing) are the same. To prevent this situation, I made test_size 0.25, which is 0.30. The accuracy score has changed. But this time I also observed that the scores decreased while the accuracy score should have increased after preprocessing. I think that the reason for this may be due to the fact that the number of features and the number of rows of my dataset are small.


















